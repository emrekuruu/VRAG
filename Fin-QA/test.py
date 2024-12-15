import boto3
import psutil
import os
import time

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing

key_folder = "../.keys" 

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


upload_directory_to_s3(".byaldi/", BUCKET_NAME, "byaldi")