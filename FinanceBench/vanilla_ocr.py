import boto3
import asyncio
import os
import time
import json
import logging
from langchain.document_loaders import PyMuPDFLoader

# Configure logging
logging.basicConfig(
    filename="text_documents.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

# AWS S3 Configuration
BUCKET_NAME = "finance-bench"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing
OUTPUT_FILE = "document_texts.json"  # Output JSON file

# Global variables
concurrency_limit = 16
semaphore = asyncio.Semaphore(concurrency_limit)
document_texts = {}  # Dictionary to store extracted text

# Define the folder path containing the keys
key_folder = "../keys"  # Replace with the correct path if needed

# Read the AWS Access Key
with open(f"{key_folder}/aws_access_key.txt", "r") as access_key_file:
    AWS_ACCESS_KEY_ID = access_key_file.read().strip()

# Read the AWS Secret Key
with open(f"{key_folder}/aws_secret_key.txt", "r") as secret_key_file:
    AWS_SECRET_ACCESS_KEY = secret_key_file.read().strip()

# Initialize boto3 client with credentials
s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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

def extract_text_from_file(local_file_path):
    """
    Extract text from a PDF using PyMuPDFLoader and format it as JSON serializable.

    Args:
        local_file_path (str): Path to the local PDF file.

    Returns:
        list: A list of dictionaries with 'page_content' and 'metadata' for each page.
    """
    try:
        pdf_reader = PyMuPDFLoader(local_file_path)
        pdf_documents = pdf_reader.load()
        
        # Format the output into a JSON serializable structure
        json_serializable_output = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in pdf_documents
        ]
        
        return json_serializable_output
    except Exception as e:
        logging.error(f"Failed to extract text from {local_file_path}: {e}")
        return None

# Define an async function to process a single file
async def process_file(file_key, document_texts):
    async with semaphore:
        # Extract metadata from the S3 key
        parts = file_key.split('_')
        company, year, type_ = parts[0], parts[1], parts[2]

        filename = company + "_" + year + "_" + type_

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
            # Extract text in a thread
            text = await asyncio.to_thread(extract_text_from_file, local_file_path)
            if text is not None:
                document_texts[file_key] = text
                logging.info(f"Successfully processed {file_key}")
            else:
                logging.error(f"Failed to extract text for {file_key}")
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

    global document_texts

    tasks = []
    batch_counter = 0
    files = list_s3_files()

    for file_key in files:
        tasks.append(asyncio.create_task(process_file(file_key, document_texts)))
        batch_counter += 1

        # Save and reset the batch every 500 files
        if batch_counter >= 50:
            await asyncio.gather(*tasks)
            tasks = []
            batch_counter = 0

            # Save the current document_texts to a JSON file
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(document_texts, f, indent=4)

            upload_file_to_s3(OUTPUT_FILE, BUCKET_NAME, "document_texts.json")

    # Process any remaining tasks in the last batch
    if tasks:
        await asyncio.gather(*tasks)

    # Final save of the document_texts dictionary
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
