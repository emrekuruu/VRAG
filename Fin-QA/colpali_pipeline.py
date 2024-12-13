import os
import pandas as pd
import asyncio
import json
import boto3
from datasets import load_dataset
from byaldi import RAGMultiModalModel
import logging
import gzip 

# Configure logging
logging.basicConfig(
    filename="colpali_retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"
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

def prepare_dataset():
    dataset = load_dataset("ibm/finqa", trust_remote_code=True)
    data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
    data.reset_index(drop=True, inplace=True)
    data = data[["id", "question", "answer", "gold_inds"]]
    data["Company"] = [row[0] for row in data.id.str.split("/")]
    data["Year"] = [row[1] for row in data.id.str.split("/")]
    data.id = data.id.map(lambda x: x.split("-")[0])
    return data 

async def process_item_qrels(data, idx, RAG):
    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Perform retrieval asynchronously
    retrieved = await asyncio.to_thread(RAG.search, query, k=5, filter_metadata={"Company_Year" : f"{company}_{year}"})

    # Construct the query's qrels
    qrels = { company + "/" + year + "/" + doc.metadata['Filename'] : doc.score for doc in retrieved}

    # Log the successful retrieval
    logging.info(f"Retrieved qrels for index {idx}")

    return idx, qrels

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
    if not os.path.exists(".byaldi/finqa"):
        print("Downloading index")
        os.mkdir(".byaldi/")
        download_s3_folder("byaldi", ".byaldi")
    else:
        print("Index already exists")

    with gzip.open(".byaldi/finqa/metadata.json.gz", "rt") as f:
        metadata = json.load(f)

    metadata = {key: {"Company_Year": value["Company"] + "_" + value["Year"], "Filename" : value["Filename"]} for key, value in metadata.items()}  

    with gzip.open(".byaldi/finqa/metadata.json.gz", "wt") as f:
        json.dump(metadata, f)

    data = prepare_dataset()

    RAG = RAGMultiModalModel.from_index(index_path="finqa", device="cuda")

    qrels = await generate_qrels(data, RAG)

    with open("results/colpali_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/qrels.json")

if __name__ == "__main__":
    asyncio.run(main())