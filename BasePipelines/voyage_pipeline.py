import asyncio
import logging
import json
from abc import ABC, abstractmethod
import os
import math
import faiss
import numpy as np
from datasets import load_dataset

class VoyageEmbedder:
    def __init__(self, voyage_client, batch_size=64):
        self.voyage_client = voyage_client
        self.batch_size = batch_size

    async def embed_query(self, query):
        response = await self.voyage_client.multimodal_embed(
            inputs=[[query]], model="voyage-multimodal-3", input_type="query"
        )
        return response.embeddings[0]

    async def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await self.voyage_client.multimodal_embed(
                inputs=[[text] for text in batch], model="voyage-multimodal-3", input_type="document"
            )
            embeddings.extend(response.embeddings)
        return embeddings


class VoyagePipeline(ABC):

    def __init__(self, config, task, index_file="faiss_index.bin", metadata_file="metadata.json"):
        self.config = config
        self.task = task
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.embedder = VoyageEmbedder(config.vo)
        self.index = None
        self.metadata = None

        logging.basicConfig(
            filename=f".logs/{self.task}-voyage_pipeline.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    async def process_query(self, data, idx):
        pass

    def load_index(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

    async def create_index(self, chunks):
        embeddings = await self.embedder.embed_documents([chunk.page_content for chunk in chunks])
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata = {i: chunk.metadata for i, chunk in enumerate(chunks)}

        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

        logging.info("FAISS index and metadata created successfully.")

    async def search(self, query_embedding, k=5, metadata_filter=None):
        if not self.index or not self.metadata:
            raise ValueError("Index and metadata must be loaded or created before searching.")

        filtered_ids = [
            doc_id for doc_id, meta in self.metadata.items()
            if not metadata_filter or all(meta.get(key) == value for key, value in metadata_filter.items())
        ]

        if not filtered_ids:
            return []

        filtered_embeddings = np.array([
            self.index.reconstruct(i) for i in filtered_ids
        ]).astype("float32")

        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = temp_index.search(query_np, k)

        results = [
            {
                "id": filtered_ids[idx],
                "distance": dist,
                "metadata": self.metadata[filtered_ids[idx]],
            }
            for dist, idx in zip(distances[0], indices[0]) if idx != -1
        ]

        return results

    async def process_queries(self, data):
        tasks = [self.process_query(data, idx) for idx in data.index]
        results = await asyncio.gather(*tasks)

        qrels = {}
        for result in results:
            qrels.update(result)

        return qrels

    async def __call__(self):
        data = self.prepare_dataset()
        self.load_index()

        qrels = await self.process_queries(data)

        with open(f"results/{self.task}_voyage_qrels.json", "w") as f:
            json.dump(qrels, f, indent=4)

        logging.info("Processing completed and results saved.")
