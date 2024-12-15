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
from Generation.generation import image_based

# Configure logging
task = "Table_VQA"

logging.basicConfig(
    filename=f"{task}/colpali_retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# AWS S3 Configuration
BUCKET_NAME = "table-vqa"
REGION_NAME = "eu-central-1"
TEMP_DIR = "/tmp/colpali_temp"  # Temporary directory for storing files

key_folder = ".keys"

with open(f"{key_folder}/aws_access_key.txt", "r") as access_key_file:
    AWS_ACCESS_KEY_ID = access_key_file.read().strip()

with open(f"{key_folder}/aws_secret_key.txt", "r") as secret_key_file:
    AWS_SECRET_ACCESS_KEY = secret_key_file.read().strip()

with open(f"{key_folder}/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

with open(f"{key_folder}/hf_key.txt", "r") as hf_key_file:
    os.environ["HUGGINGFACE_API_KEY"] = hf_key_file.read().strip()

s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

aws_semaphore = asyncio.Semaphore(10)  
qrel_semaphore = asyncio.Semaphore(16)

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

async def fetch_file_as_base64_images(company, year, filename):

    file_key = f"{company}/{year}/{filename}"
    local_file_path = os.path.join(TEMP_DIR, filename)

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

    async with qrel_semaphore: 

        query = data.iloc[idx]["question"]
        company = data.iloc[idx]["Company"]
        year = data.iloc[idx]["Year"]

        retrieved = await asyncio.to_thread(RAG.search, query, k=top_n, filter_metadata={"Company_Year": f"{company}_{year}"})

        # Fetch Base64-encoded images
        pages = []
        file_keys = []

        for doc in retrieved:
            filename = doc.metadata["Filename"]
            file_keys.append(f"{company}/{year}/{filename}")
            base64_images = await fetch_file_as_base64_images(company, year, filename)
            if base64_images:
                pages.extend(base64_images) 

        if not pages:
            logging.warning(f"No image documents prepared for query index {idx}: {query}")

        if reranker is None:
            qrels = { company + "/" + year + "/" + doc.metadata['Filename'] : doc.score for doc in retrieved}

        else:
            results = await asyncio.to_thread(reranker.rank, query, pages)
            qrels = { company + "/" + year + "/" + retrieved[doc.doc_id]["metadata"]["Filename"] : doc.score for doc in results.top_k(top_k)}

        answer = await image_based(query, pages)
            
        return idx, qrels, answer


async def generate_qrels(data, RAG, reranker, top_n):
    """
    Generates qrels for the entire dataset, including reranking.
    """
    qrels = {}
    answers = {}

    # Create tasks for processing each item
    tasks = [process_item_qrels(data, idx, RAG, reranker, top_n) for idx in data.index]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the qrels dictionary
    for query_id, retrieved_qrels, answer in results_list:
        qrels[query_id] = retrieved_qrels
        answers[query_id] = answer

    return qrels, answers

async def main():

    if not os.path.exists(f"{task}/.byaldi/table_vqa"):
        print("Downloading index")
        os.makedirs(".byaldi/", exist_ok=True)
    else:
        print("Index already exists")

    data = prepare_dataset()

    data = data.iloc[0:2]

    top_n = 5

    RAG = RAGMultiModalModel.from_index(index_path=f"/workspace/VRAG/{task}/.byaldi/table_vqa", device="cuda")

    # reranker = Reranker("monovlm", device="cuda")
    reranker = None

    qrels, answers = await generate_qrels(data, RAG, reranker, top_n)

    with open(f"{task}/results/colpali/colpali_{top_n}_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    with open(f"{task}/results/generation/image_answers.json", "w") as f:
        json.dump(answers, f, indent=4)

    print("Finished")

if __name__ == "__main__":
    asyncio.run(main())
