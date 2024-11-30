import os
import pandas as pd 
import asyncio
import gzip
import json
import boto3
from datasets import load_dataset
from byaldi import RAGMultiModalModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import logging

# Configure logging
logging.basicConfig(
    filename="retrieval.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(16)

key_folder = "../keys"

with open('../keys/hf_key.txt', 'r') as file:
    hf_key = file.read().strip()

with open("../keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()

os.environ["HF_TOKEN"] = hf_key
os.environ["OPENAI_API_KEY"] = openai_key

# AWS S3 Configuration
BUCKET_NAME = "finance-bench"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing

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


def prepare_dataset():
    data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
    data[['Company', 'Year', 'Type']] = data['doc_name'].str.split('_', expand=True).iloc[:, :3]
    return data


async def process_item(data, idx, RAG):

    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]
    type_ = data.loc[idx, "Type"]
    
    if company == "JOHNSON":
        company = "JOHNSON & JOHNSON"
        year = data.loc[idx, 'doc_name'].str.split('_', expand=True)[3]
        type_ = data.loc[idx, 'doc_name'].str.split('_', expand=True)[4]

    # Perform retrieval asynchronously
    try:
        retrieved = await asyncio.to_thread(RAG.search, query, k=1, filter_metadata={"Company": company, "Year": year, "Type": type_ + ".pdf"})
    except Exception as e:
        print(data.loc[idx, "question"])
        print(company, year, type_)
        return idx, []
    
    # Populate the results
    retrieved_context = f"{company}/{year}/{type_}/{retrieved[0].page_num}"

    # Log the successful retrieval
    logging.info(f"Retrieved context for index {idx}")

    return idx, retrieved_context


async def generate(data, RAG):

    # Initialize the results DataFrame
    results = pd.DataFrame(columns=["Retrieved Context"], index=data.index)

    # Create tasks for processing each item
    tasks = [process_item(data, idx, RAG) for idx in data.index]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the results DataFrame
    for idx, retrieved_context in results_list:
        results.loc[idx, "Retrieved Context"] = retrieved_context

    return results

def download_s3_folder(prefix, local_dir):
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

async def main():

    if not os.path.exists(".byaldi/finance_bench"):
        print("Downloading index")
        os.mkdir(".byaldi/")
        download_s3_folder("byaldi", ".byaldi")
    else:
        print("Index already exists")

    data = prepare_dataset()
    
    RAG = RAGMultiModalModel.from_index(index_path="finance_bench", device="mps")

    results = await generate(data, RAG)
    results.to_csv("results/colpali.csv", index=True)
    
if __name__ == "__main__":
    asyncio.run(main())