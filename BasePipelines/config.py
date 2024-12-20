import os
import boto3
import asyncio
import logging
import time
import base64
from io import BytesIO
from pdf2image import convert_from_bytes
import voyageai
import cohere

class Config:
    """
    A configuration class to centralize API key management, AWS S3 client setup,
    and utility methods for AWS S3 operations.
    """

    def __init__(self, bucket_name, key_folder=".keys", region_name="eu-central-1", temp_dir="/tmp"):
        """
        Initialize the configuration and set up API keys, AWS S3 client, and utility attributes.

        Args:
            bucket_name (str): Name of the S3 bucket.
            key_folder (str): Path to the folder containing API keys.
            region_name (str): AWS region for the S3 client.
            temp_dir (str): Directory for temporary files.
        """
        self.bucket_name = bucket_name
        self.key_folder = key_folder
        self.region_name = region_name
        self.temp_dir = temp_dir

        # AWS-related attributes
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.s3_client = None

        # Initialize everything
        self.setup_api()
        self.setup_aws_s3()

    def setup_api(self):
        """Load API keys and set environment variables."""
        with open(f"{self.key_folder}/openai_api_key.txt", "r") as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()

        with open(f"{self.key_folder}/hf_key.txt", "r") as file:
            os.environ["HUGGINGFACE_API_KEY"] = file.read().strip()

        with open(f"{self.key_folder}/aws_access_key.txt", "r") as file:
            self.aws_access_key_id = file.read().strip()

        with open(f"{self.key_folder}/aws_secret_key.txt", "r") as file:
            self.aws_secret_access_key = file.read().strip()

        with open(f"{self.key_folder}/voyage_api_key.txt",  "r") as file:
            voyage_api_key = file.read().strip()

        with open(f"{self.key_folder}/cohere_api_key.txt",  "r") as file:
            cohere_api_key = file.read().strip()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.vo = voyageai.Client(api_key=voyage_api_key)
        self.co = cohere.Client(api_key=cohere_api_key)

    def setup_aws_s3(self):
        """Initialize the AWS S3 client."""
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS keys not initialized. Call setup_api() first.")

        self.s3_client = boto3.client(
            's3',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def list_s3_files(self, prefix=""):
        """
        List all files in the S3 bucket with a given prefix.

        Args:
            prefix (str): Prefix to filter the files.

        Returns:
            list: List of file keys in the S3 bucket.
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        files = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            files.extend([obj['Key'] for obj in page.get('Contents', [])])
        return files

    def download_s3_folder(self, prefix, local_dir):
        """
        Download an entire folder from S3 to a local directory.

        Args:
            prefix (str): Prefix of the folder in the S3 bucket.
            local_dir (str): Local directory to download the files to.
        """
        os.makedirs(local_dir, exist_ok=True)
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                file_key = obj['Key']
                if file_key.endswith('/'):
                    continue

                relative_path = os.path.relpath(file_key, prefix)
                local_file_path = os.path.join(local_dir, relative_path)

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                self.s3_client.download_file(self.bucket_name, file_key, local_file_path)

    async def fetch_file_as_base64_images(self, file_key, filename, semaphore, max_wait_time=20, page = None):
        """
        Fetch a file from S3, convert it to Base64 images, and return the images.

        Args:
            file_key (str): Key of the file in S3.
            filename (str): Name of the file for local storage.
            semaphore (asyncio.Semaphore): Semaphore for concurrency control.
            max_wait_time (int): Maximum wait time for file availability.

        Returns:
            list: List of Base64-encoded images, or None if an error occurred.
        """
        local_file_path = os.path.join(self.temp_dir, filename)
        os.makedirs(self.temp_dir, exist_ok=True)

        async with semaphore:
            try:
                self.s3_client.download_file(self.bucket_name, file_key, local_file_path)

                elapsed_time = 0
                while not os.path.exists(local_file_path):
                    if elapsed_time >= max_wait_time:
                        logging.error(f"File did not appear within {max_wait_time} seconds: {local_file_path}")
                        return None
                    time.sleep(0.5)
                    elapsed_time += 0.5

                with open(local_file_path, "rb") as file:
                    pdf_bytes = file.read()

                    if page is not None:
                        images = convert_from_bytes(pdf_bytes, fmt="jpeg", dpi=200, first_page=page, last_page=page)
                    else:
                        images = convert_from_bytes(pdf_bytes, fmt="jpeg", dpi=200)

                base64_images = []
                for img in images:
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    base64_images.append(img_base64)

                return base64_images
            except Exception as e:
                logging.error(f"Failed to fetch and convert file {file_key} to Base64 images: {e}")
                return None
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
