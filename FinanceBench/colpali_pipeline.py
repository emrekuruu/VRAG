import os
import pandas as pd
import asyncio
import json
import base64
import boto3
from datasets import load_dataset
from byaldi import RAGMultiModalModel
from rerankers import Reranker
import logging
import time
import torch 
from pdf2image import convert_from_bytes
from langchain_core.documents import Document
from io import BytesIO


# Configure logging
logging.basicConfig(
    filename="colpali_retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# AWS S3 Configuration
BUCKET_NAME = "finance-bench"
REGION_NAME = "eu-central-1"
TEMP_DIR = "/tmp/docs/temp"

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

aws_semaphore = asyncio.Semaphore(10)  
qrel_semaphore = asyncio.Semaphore(16)

def prepare_dataset():
    data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
    data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
    return data 

async def fetch_file_as_base64_images(file_key, page=None):
    local_file_path = os.path.join(TEMP_DIR, file_key)

    # Create the temporary directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)

    async with aws_semaphore:
        try:
            # Download the file from S3
            s3.download_file(BUCKET_NAME, file_key, local_file_path)

            # Wait until the file is fully downloaded
            max_wait_time = 20  # seconds
            elapsed_time = 0
            while not os.path.exists(local_file_path):
                if elapsed_time >= max_wait_time:
                    logging.error(f"File did not appear within {max_wait_time} seconds: {local_file_path}")
                    return None
                time.sleep(0.5)
                elapsed_time += 0.5

            # Convert PDF to images
            with open(local_file_path, "rb") as file:
                pdf_bytes = file.read()

                # Extract a single page if specified
                if page is not None:
                    images = convert_from_bytes(pdf_bytes, fmt="jpeg", dpi=200, first_page=page, last_page=page)
                else:
                    # Convert all pages if no specific page is specified
                    images = convert_from_bytes(pdf_bytes, fmt="jpeg", dpi=200)

            # Convert images to Base64
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
            # Remove the temporary file to free up space
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

async def process_item_qrels(data, idx, RAG, reranker, top_n=10, top_k=5):
    """
    Processes a single query to retrieve and rerank qrels, using Base64-encoded image inputs.
    """
    async with qrel_semaphore: 

        query = data.loc[idx, "question"]
        doc_name = data.loc[idx, "doc_name"] + ".pdf"

        # Perform retrieval asynchronously
        retrieved = await asyncio.to_thread(RAG.search, query, k=top_n, filter_metadata={"Filename" : doc_name})

        if reranker is None:
                qrels = { company + "/" + year + "/" + doc.metadata['Filename'] : doc.score for doc in retrieved}
                

        else:

            # Fetch Base64-encoded images
            passages = []
            file_keys = []

            for doc in retrieved:
                filename = doc.metadata["Filename"] 
                page = int(doc.page_num)
                file_keys.append(f"{filename}")
                base64_images = await fetch_file_as_base64_images(filename, page)
                if base64_images:
                    passages.extend(base64_images)  

            if not passages:
                logging.warning(f"No image documents prepared for query index {idx}: {query}")
                return idx, {}

            # Perform reranking
            results = await asyncio.to_thread(reranker.rank, query, passages)

            qrels = { doc_name + "_page_" +  str(retrieved[doc.doc_id].page_num - 1 ) : doc.score for doc in results.top_k(top_k)}

        logging.info(f"Successfully retrieved qrels for query index {idx}")
            
        return idx, qrels


async def generate_qrels(data, RAG, reranker, top_n):

    qrels = {}

    # Create tasks for processing each item
    tasks = [process_item_qrels(data, idx, RAG, reranker, top_n) for idx in data.index]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the qrels dictionary
    for query_id, retrieved_qrels in results_list:
        qrels[query_id] = retrieved_qrels

    return qrels

def download_s3_folder(prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_key = obj['Key']
            if file_key.endswith('/'):
                continue
            relative_path = os.path.relpath(file_key, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
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

    top_n = 20

    RAG = RAGMultiModalModel.from_index(index_path="finance_bench", device="cuda")

    reranker = Reranker("monovlm", device="cuda")

    qrels = await generate_qrels(data, RAG, reranker, top_n)

    os.makedirs("results/colpali", exist_ok=True)
    with open(f"results/colpali/colpali_{top_n}_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/colpali")

if __name__ == "__main__":
    asyncio.run(main())
