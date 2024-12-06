import os
import boto3
import asyncio
import logging
import voyageai
import fitz
import faiss
import numpy as np
from PIL import Image
import json 

logging.basicConfig(
    filename="chunking.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

semaphore = asyncio.Semaphore(16)

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

def download_s3_folder(prefix, local_dir):
    """
    Download all files from an S3 prefix to a local directory, skipping directories.
    """
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

def list_s3_files(prefix=""):
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        files.extend([obj['Key'] for obj in page.get('Contents', [])])
    return files

with open("../keys/voyage_api_key.txt", "r") as file:
    voyage_api_key = file.read().strip()

vo = voyageai.AsyncClient(api_key=voyage_api_key)

class Embedder:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def pdf_to_image(self, pdf_path, zoom=1.0):
        """Convert the first page of a PDF into an image."""
        pdf_document = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_document.close()
        return img

    async def embed_document(self, file_path):
        """Asynchronously embed a document represented by its file path."""
        image = self.pdf_to_image(file_path)
        embedding_obj = await vo.multimodal_embed(
            inputs=[[image]],
            model="voyage-multimodal-3",
            input_type="document"
        )
        return embedding_obj.embeddings[0]

class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}

    def add(self, ids, embeddings, metadatas):
        """Add embeddings and metadata."""
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        for i, doc_id in enumerate(ids):
            self.metadata[doc_id] = metadatas[i]

    def search(self, query_embedding, k=5):
        """Search the vector store."""
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_np, k)
        results = [
            {
                "id": list(self.metadata.keys())[idx],
                "distance": dist,
                "metadata": self.metadata[list(self.metadata.keys())[idx]],
            }
            for dist, idx in zip(distances[0], indices[0])
        ]
        return results

async def process_file(file_key, vector_store, embedder):
    async with semaphore:
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]
        local_file_path = f"/tmp/{filename}"
        
        s3.download_file(BUCKET_NAME, file_key, local_file_path)
        
        try:
            embedding = await embedder.embed_document(local_file_path)
            
            metadata = {
                "Company": company,
                "Year": year,
                "Filename": filename,
            }

            vector_store.add(
                ids=[file_key],
                embeddings=[embedding],
                metadatas=[metadata],
            )

            logging.info(f"Successfully added PDF {company}/{year}/{filename} to FAISS")

        except Exception as e:
            logging.error(f"Failed to process file {file_key}: {e}")

        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

async def process_all(vector_store, embedder, batch_size=1000):
    tasks = []
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
                tasks.append(asyncio.create_task(process_file(file_key, vector_store, embedder)))

    total_tasks = len(tasks)
    for i in range(0, total_tasks, batch_size):
        batch_tasks = tasks[i:i + batch_size]
        await asyncio.gather(*batch_tasks)
        logging.info(f"Processed {i + len(batch_tasks)} out of {total_tasks} files")

async def main():

    embedder = Embedder()
    vector_store = FAISSVectorStore(dimension=1024) 
    await process_all(vector_store, embedder)
    faiss.write_index(vector_store.index, "faiss_index.bin")

    with open("metadata.json", "w") as f:
        json.dump(vector_store.metadata, f)

    logging.info("All files have been processed and added to FAISS vector store.")

if __name__ == "__main__":
    asyncio.run(main())
