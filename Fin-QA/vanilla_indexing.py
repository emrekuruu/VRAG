import os
import time
import boto3
import asyncio
import pandas as pd
import pickle
from io import StringIO
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import logging 

logging.basicConfig(
    filename="chunking.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(16)

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

def list_s3_files(prefix=""):
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        files.extend([obj['Key'] for obj in page.get('Contents', [])])
    return files

def process_table(chunk):
    table = pd.read_html(StringIO(chunk.metadata.text_as_html))[0]
    table.set_index(0, inplace=True)
    table.columns = table.columns.astype(str) 
    table = table.astype(str)

    if table.index.name == 0:
        table.index.name = None

    while pd.isna(table.index.values[0]):
        table.columns = table.iloc[0]
        table = table.iloc[1:]

    return table.to_string()


async def process_file(file_key):

    documents = {file_key: []}

    async with semaphore:
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]

        local_file_path = f"/tmp/{filename}"
        
        try:
            s3.download_file(BUCKET_NAME, file_key, local_file_path)
        except Exception as e:
            print(f"Failed to download {file_key} from S3: {e}")
            return

        # Wait for the file to exist in /tmp
        max_wait_time = 20  # Maximum wait time in seconds
        elapsed_time = 0
        while not os.path.exists(local_file_path):
            if elapsed_time >= max_wait_time:
                print(f"File did not appear in /tmp within {max_wait_time} seconds: {local_file_path}")
                return
            time.sleep(0.5)
            elapsed_time += 0.5

        try:
            elements = await asyncio.to_thread(partition_pdf, filename=local_file_path, strategy="hi_res", infer_table_structure=True)

            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=1500, 
                max_characters=int(1e6),          
            )        

            for i, chunk in enumerate(chunks):

                if len(chunk.text) < 10:
                    chunks.remove(chunk)

                if chunk.category == "Table":
                    
                    try:
                        previous_chunk = chunks[i-1]
                        caption = previous_chunk.text.split("\n")[-1]
                        previous_chunk.text = previous_chunk.text.replace(caption, "")
                        
                        if len(previous_chunk.text) < 50:
                            chunks.remove(previous_chunk)
                        
                        chunk.text = f"Table Caption: ** {caption}**\n\n " + process_table(chunk)
                    except:
                        chunk.text = process_table(chunk)

            
            for chunk in chunks:
                documents[file_key].append(Document(page_content=chunk.text, metadata={"category": chunk.category, "Company": company, "Year": year, "Filename": filename}))

        except Exception as e:
            print(f"Failed to extract text from {file_key}: {e}")

        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

        return documents

async def process_all(batch_size=10):
    tasks = []
    docs = {}

    companies = list_s3_files()
    companies = set([key.split('/')[0] for key in companies if '/' in key])

    for company in companies:

        if company == "byaldi":
            continue    
        
        years = list_s3_files(f"{company}/")
        years = set([key.split('/')[1] for key in years if '/' in key])

        for year in years:
            files = list_s3_files(f"{company}/{year}/")

            for file_key in files:
                tasks.append(asyncio.create_task(process_file(file_key)))

    total_tasks = len(tasks)
    for i in range(0, total_tasks, batch_size):
        batch_tasks = tasks[i:i + batch_size]

        batch_results = await asyncio.gather(*batch_tasks)
        batch_results = [result for result in batch_results if result is not None]

        for result in batch_results:
            if result:
                docs.update(result)

        with open("processed_documents.pkl", "wb") as f:
            pickle.dump(docs, f)

        logging.info(f"Processed {i + len(batch_results)} out of {total_tasks} documents")

    return docs


async def main():

    output_file = "processed_documents.pkl"

    with open(output_file, "wb") as file:
        pass 

    logging.info("Starting document processing...")
    await process_all()

    logging.info(f"Document processing completed. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
