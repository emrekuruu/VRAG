import boto3
import asyncio
import psutil
import os
import time
from byaldi import RAGMultiModalModel

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing

# Global variables
concurrency_limit = 8
semaphore = asyncio.Semaphore(concurrency_limit)

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"
REGION_NAME = "eu-central-1"
TEMP_DIR = "/tmp/docs/temp"

key_folder = "../keys" 

with open(f"{key_folder}/aws_access_key.txt", "r") as access_key_file:
    AWS_ACCESS_KEY_ID = access_key_file.read().strip()

with open(f"{key_folder}/aws_secret_key.txt", "r") as secret_key_file:
    AWS_SECRET_ACCESS_KEY = secret_key_file.read().strip()
    
s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

RAG = None

# Function to download files from S3 to a local directory
def download_s3_folder(prefix, local_dir):
    """
    Download all files from an S3 prefix to a local directory, skipping directories.
    """
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_key = obj['Key']
            
            # Skip keys that are treated as directories
            if file_key.endswith('/'):
                continue

            # Preserve directory structure
            relative_path = os.path.relpath(file_key, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Create subdirectories as needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            s3.download_file(BUCKET_NAME, file_key, local_file_path)

# Define an async function to process a single file
async def process_file(file_key):
    async with semaphore:
        global RAG

        # Extract metadata from the S3 key
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]

        # Temporary local file path
        local_file_path = f"/tmp/{filename}"
        
        # Download file from S3
        s3.download_file(BUCKET_NAME, file_key, local_file_path)

        # Process the downloaded file with RAG
        await asyncio.to_thread(
            RAG.add_to_index,
            input_item=local_file_path,  # Use the local file path here
            store_collection_with_index=True,
            metadata={"Company": company, "Year": year, "Filename": filename},
        )

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
    companies = ["AAL"]

    for company in companies:
        years = list_s3_files(f"{company}/")
        years = set([key.split('/')[1] for key in years if '/' in key])
        years = ["2014"]

        for year in years:
            files = list_s3_files(f"{company}/{year}/")

            for file_key in files:
                tasks.append(asyncio.create_task(process_file(file_key)))

    await asyncio.gather(*tasks)


def upload_directory_to_s3(local_dir, bucket, s3_prefix):
    """
    Recursively upload a local directory to S3.

    Args:
        local_dir (str): Path to the local directory.
        bucket (str): Name of the S3 bucket.
        s3_prefix (str): S3 prefix where files will be uploaded.
    """
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create the relative S3 path
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure POSIX paths for S3
            print(f"Uploading {local_file_path} to s3://{bucket}/{s3_key}")
            s3.upload_file(local_file_path, bucket, s3_key)

# Main async function
async def main():
    global RAG

    # Step 1: Initialize the RAG model
    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", device="cuda")

    # Step 2: Initialize the index using the temp folder
    print("Initializing the index with the temp folder...")
    download_s3_folder(prefix="temp/", local_dir=TEMP_DIR)
    RAG.index(
        input_path=TEMP_DIR,
        index_name="finqa",
        overwrite=True,
    )
    print("Index initialized successfully.")

    # Step 3: Process all files in the bucket
    print("Processing all files in the bucket...")
    start_time = time.time()
    await process_all()
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    print("Uploading the .byaldi index directory to S3...")
    index_dir_path = os.path.join(os.getcwd(), ".byaldi")  # Current working directory

    if os.path.exists(index_dir_path):
        upload_directory_to_s3(index_dir_path, BUCKET_NAME, "byaldi")
        print("Index directory uploaded successfully to S3.")

    else:
        print(f"Index directory not found in the current working directory: {index_dir_path}.")

if __name__ == "__main__":
    asyncio.run(main())
