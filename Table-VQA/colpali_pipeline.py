import os
import pandas as pd
import asyncio
import gzip
import json
import boto3
from datasets import load_dataset
from byaldi import RAGMultiModalModel
from rerankers import Reranker
import logging

# Configure logging
logging.basicConfig(
    filename="colpali_retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# AWS S3 Configuration
BUCKET_NAME = "table-vqa"
REGION_NAME = "eu-central-1"
TEMP_DIR = "/tmp/docs/temp"

key_folder = "../.keys" 

with open(f"{key_folder}/aws_access_key.txt", "r") as access_key_file:
    AWS_ACCESS_KEY_ID = access_key_file.read().strip()

with open(f"{key_folder}/aws_secret_key.txt", "r") as secret_key_file:
    AWS_SECRET_ACCESS_KEY = secret_key_file.read().strip()

with open(f"{key_folder}/hf_key.txt", "r") as hf_key_file:
    os.environ["HUGGINGFACE_API_KEY"] = hf_key_file.read().strip()
    
s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def prepare_dataset():

    def process_qa_id(qa_id):
        splitted = qa_id.split(".")[0]
        return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

    data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt"]]
    data.qa_id = data.qa_id.apply(process_qa_id)
    data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
    data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
    data = data.rename(columns={"qa_id": "id"})
    return data

async def process_item_qrels(data, idx, RAG, reranker, top_n=10, top_k=5):
    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Perform retrieval asynchronously
    retrieved = await asyncio.to_thread(RAG.search, query, k=top_n, filter_metadata={"Company": company, "Year": year})

    # Prepare inputs for reranking
    passages = [doc.content for doc in retrieved]
    scores = [doc.score for doc in retrieved]

    # Perform reranking using MonoQwen
    reranked_scores = await asyncio.to_thread(reranker, query=query, passages=passages)

    # Select top_k results after reranking
    top_k_indices = sorted(range(len(reranked_scores)), key=lambda i: reranked_scores[i], reverse=True)[:top_k]
    reranked_results = {
        company + "/" + year + "/" + retrieved[i].metadata['Filename']: reranked_scores[i] for i in top_k_indices
    }

    # Log the successful retrieval and reranking
    logging.info(f"Retrieved and reranked qrels for index {idx}")

    return idx, reranked_results

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
    if not os.path.exists(".byaldi/table_vqa"):
        print("Downloading index")
        os.mkdir(".byaldi/")
        download_s3_folder("byaldi", ".byaldi")
    else:
        print("Index already exists")

    data = prepare_dataset()

    top_n = 10

    RAG = RAGMultiModalModel.from_index(index_path="table_vqa", device="mps")

    # Initialize the reranker with MonoQwen
    reranker = Reranker("monovlm",device="mps")

    # Generate qrels with reranking
    qrels = await generate_qrels(data, RAG, reranker, top_n)

    # Save qrels to a JSON file for later use
    with open("results/colpali/colpali_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/qrels.json")

if __name__ == "__main__":
    asyncio.run(main())
