import boto3
import asyncio
import psutil
import os
import time
import json
from pdf2image import convert_from_path
import pytesseract
import logging

# Configure logging
logging.basicConfig(
    filename="retrieval.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing
OUTPUT_FILE = "/tmp/docs/document_texts.json"  # Output JSON file

# Global variables
concurrency_limit = 16
semaphore = asyncio.Semaphore(concurrency_limit)
document_texts = {}  # Dictionary to store extracted text

# Initialize boto3 client with credentials
s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

def upload_file_to_s3(local_file, bucket, s3_key):
    """
    Upload a file to S3.

    Args:
        local_file (str): Path to the local file.
        bucket (str): Name of the S3 bucket.
        s3_key (str): S3 key where the file will be saved.
    """
    try:
        print(f"Uploading {local_file} to s3://{bucket}/{s3_key}...")
        s3.upload_file(local_file, bucket, s3_key)
        print(f"File uploaded successfully to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload {local_file} to S3: {e}")

# Define an async function to process a single file
async def process_file(file_key):
    async with semaphore:
        # Extract metadata from the S3 key
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]

        # Temporary local file path
        local_file_path = f"/tmp/{filename}"
        
        # Download file from S3
        try:
            s3.download_file(BUCKET_NAME, file_key, local_file_path)
        except Exception as e:
            print(f"Failed to download {file_key} from S3: {e}")
            return

        # Wait for the file to exist in /tmp
        max_wait_time = 10  # Maximum wait time in seconds
        elapsed_time = 0
        while not os.path.exists(local_file_path):
            if elapsed_time >= max_wait_time:
                print(f"File did not appear in /tmp within {max_wait_time} seconds: {local_file_path}")
                return
            time.sleep(0.5)
            elapsed_time += 0.5

        # Extract text from the document
        try:
            pages = convert_from_path(local_file_path)
            for page_num, page_image in enumerate(pages, start=1):
                text = pytesseract.image_to_string(page_image)
                key = f"{company}/{year}/page_{page_num}"
                document_texts[key] = text
                logging.info(f"Done with {key}")
        except Exception as e:
            print(f"Failed to extract text from {file_key}: {e}")
        finally:
            # Clean up the local file after processing
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

# Function to list all files under a prefix in the S3 bucket
def list_s3_files(prefix=""):
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        files.extend([obj['Key'] for obj in page.get('Contents', [])])
    return files

# Async function to process all files in the bucket
async def process_all():
    tasks = []

    companies = list_s3_files()
    companies = set([key.split('/')[0] for key in companies if '/' in key])

    for company in companies:
        years = list_s3_files(f"{company}/")
        years = set([key.split('/')[1] for key in years if '/' in key])

        for year in years:
            files = list_s3_files(f"{company}/{year}/")

            for file_key in files:
                tasks.append(asyncio.create_task(process_file(file_key)))

    await asyncio.gather(*tasks)

    # Save the document texts dictionary to a JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(document_texts, f, indent=4)

    upload_file_to_s3(OUTPUT_FILE, BUCKET_NAME, "document_texts.json")

# Main async function
async def main():

    print("Processing all files in the bucket...")
    start_time = time.time()
    await process_all()
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
