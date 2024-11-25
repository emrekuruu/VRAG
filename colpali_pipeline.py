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

device = "mps"

with open('keys/hf_key.txt', 'r') as file:
    hf_key = file.read().strip()

with open("keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()

os.environ["HF_TOKEN"] = hf_key
os.environ["OPENAI_API_KEY"] = openai_key

# AWS S3 Configuration
BUCKET_NAME = "colpali-docs"  # Replace with your bucket name
REGION_NAME = "eu-central-1"  # Replace with your bucket's region
TEMP_DIR = "/tmp/docs/temp"  # Temporary local directory for indexing

s3 = boto3.client(
    's3',
    region_name=REGION_NAME,
    aws_access_key_id="",
    aws_secret_access_key=""
)

def prepare_dataset():

    # Load the dataset
    dataset = load_dataset("ibm/finqa", trust_remote_code=True)

    # Access the splits
    data = dataset['train'].to_pandas()
    validation_data = dataset['validation'].to_pandas()
    test_data = dataset['test'].to_pandas()

    data = pd.concat([data, validation_data, test_data])
    data.reset_index(drop=True, inplace=True)
    data = data[["id", "question", "answer", "gold_inds"]]

    data["Company"] = [row[0] for row in data.id.str.split("/")]
    data["Year"] = [row[1] for row in data.id.str.split("/")]

    return data


async def process_item(data, idx, RAG):

    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Perform retrieval asynchronously
    retrieved = await asyncio.to_thread(RAG.search, query, k=1, filter_metadata={"Company": company, "Year": year})

    # Populate the results
    retrieved_context = f"{company}/{year}/{retrieved[0].metadata['Page']}"

    return idx, retrieved_context


async def generate(data, RAG, model, image_prompt):

    # Initialize the results DataFrame
    results = pd.DataFrame(columns=["Retrieved Context"], index=data.index)

    # Create tasks for processing each item
    tasks = [process_item(data, idx, RAG, model, image_prompt) for idx in data.index]

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

    if not os.path.exists(".byaldi/finqa"):
        print("Downloading index")
        os.mkdir(".byaldi/")
        download_s3_folder("byaldi", ".byaldi")
    else:
        print("Index already exists")

    data = prepare_dataset()

    unique_companies = set(data.Company.unique())

    needed_years = {}

    for company in unique_companies:
        needed_years[company] = list(data[data.Company == company].Year.unique())

    unique_companies = set(data.Company.unique())

    needed_years = {}

    files = []

    for company in unique_companies:
        needed_years[company] = list(data[data.Company == company].Year.unique())

    for company in needed_years.keys():
        for year in needed_years[company]:
            try:
                for page in os.listdir(f"docs/{company}/{year}/"):
                    files.append(f"docs/{company}/{year}/{page}")
            except:
                print(f"docs/{company}/{year}/")
                
    files = [file[5:] for file in files]

    RAG = RAGMultiModalModel.from_index(index_path="finqa", device="mps")

    missing_files = []

    with gzip.open('finqa/metadata.json.gz', 'rt') as file:
        files_ = file.read()
        files_ = json.loads(files_)

    company_names = set({f'{value["Company"]}/{value["Year"]}/{value["Filename"]}' for value in files_.values()})

    for file in files:
        if file not in company_names:
            missing_files.append(file)

    if len(missing_files) > 0:
        print(f"Missing {len(missing_files)} files, now adding to index")
        for file in missing_files:
            company, year, page = file.split("/")
            RAG.add_to_index(f"docs/{company}/{year}/{page}",store_collection_with_index=True, metadata={"Company": company, "Year": year, "Filename": page})
    
    results = await generate(data, RAG)
    results.to_csv("results/colpali.csv", index=False)
    
if __name__ == "__main__":
    asyncio.run(main())