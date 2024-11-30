import os
import pandas as pd
import asyncio
import gzip
import json
import boto3
from datasets import load_dataset
from byaldi import RAGMultiModalModel
import logging

# Configure logging
logging.basicConfig(
    filename="retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# AWS S3 Configuration
BUCKET_NAME = "table-vqa"
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

async def process_item_qrels(data, idx, RAG):
    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Perform retrieval asynchronously
    retrieved = await asyncio.to_thread(RAG.search, query, k=5, filter_metadata={"Company": company, "Year": year})

    # Construct the query's qrels
    qrels = { company + "/" + year + "/" + doc.metadata['Filename'] : doc.score for doc in retrieved}

    # Log the successful retrieval
    logging.info(f"Retrieved qrels for index {idx}")

    return data.loc[idx, "id"], qrels

async def generate_qrels(data, RAG):
    qrels = {}

    # Create tasks for processing each item
    tasks = [process_item_qrels(data, idx, RAG) for idx in data.index]

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
    data.id = data.id.map(lambda x: x.split("-")[0])

    RAG = RAGMultiModalModel.from_index(index_path="table_vqa", device="cpu")

    # Generate qrels
    qrels = await generate_qrels(data, RAG)

    # Save qrels to a JSON file for later use
    with open("results/qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/qrels.json")

if __name__ == "__main__":
    asyncio.run(main())
