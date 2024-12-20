import asyncio
import logging
import os
from abc import ABC, abstractmethod
import faiss
import numpy as np
import boto3
from PIL import Image
import fitz
import json

class VoyageEmbedder:
    def __init__(self,vo):
        self.vo = vo

    def pdf_to_images(self, pdf_path, zoom=1.0):
        pdf_document = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_document.close()
        return images

    async def embed_document(self, file_path):
        images = self.pdf_to_images(file_path)
        embeddings = []
        for image in images:
            embedding_obj = await self.vo.multimodal_embed(
                inputs=[[image]],
                model="voyage-multimodal-3",
                input_type="document"
            )
            embeddings.append(embedding_obj.embeddings[0])
        return embeddings

class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}

    def add(self, ids, embeddings, metadatas):
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        for i, doc_id in enumerate(ids):
            self.metadata[doc_id] = metadatas[i]

class VoyageIndexing(ABC):

    def __init__(self, config, task, dimension=1024, temp_dir="/tmp/docs/temp", concurrency_limit=8):
        self.config = config
        self.task = task
        self.dimension = dimension
        self.temp_dir = temp_dir
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.vector_store = self.initialize_vector_store()
        self.embedder = self.initialize_embedder()

        logging.basicConfig(
            filename=f".logs/{task}-voyage-indexing.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def initialize_vector_store(self):
        return FAISSVectorStore(dimension=self.dimension)

    def initialize_embedder(self):
        return VoyageEmbedder(vo=self.config.vo)

    @abstractmethod
    def prepare_metadata(self, file_key, page_number):
        pass

    async def embed_file(self, file_key):
        async with self.semaphore:
            local_file_path = os.path.join(self.temp_dir, os.path.basename(file_key))
            await asyncio.to_thread(
                self.config.s3_client.download_file,
                self.config.bucket_name,
                file_key,
                local_file_path,
            )
            try:
                embeddings = await self.embedder.embed_document(local_file_path)
                for page_number, embedding in enumerate(embeddings):
                    metadata = self.prepare_metadata(file_key)
                    self.vector_store.add(
                        ids=[f"{file_key}_page_{page_number}"],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                logging.info(f"Processed and added file: {file_key} with {len(embeddings)} pages.")
            except Exception as e:
                logging.error(f"Error processing file {file_key}: {e}")
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)

    async def process_files(self, prefix=""):
        file_keys = self.config.list_s3_files(prefix=prefix)
        tasks = [self.embed_file(file_key) for file_key in file_keys]
        await asyncio.gather(*tasks)

    async def save_index(self):
        faiss.write_index(self.vector_store.index, f"{self.task}_faiss_index.bin")
        with open(f"{self.task}_metadata.json", "w") as f:
            json.dump(self.vector_store.metadata, f)
        logging.info("Index and metadata saved successfully.")

    async def run(self, prefix=""):
        os.makedirs(self.temp_dir, exist_ok=True)
        await self.process_files(prefix=prefix)
        await self.save_index()
