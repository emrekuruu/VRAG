import asyncio
import logging
import json
from abc import ABC, abstractmethod
import os
import math
import faiss
import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

class FaissWithMetadata:
    def __init__(self, index_file, metadata_file):
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

    def search(self, query_embedding, k=5, metadata_filter=None):
        # Filter metadata first
        filtered_ids = [
            doc_id for doc_id, meta in self.metadata.items()
            if not metadata_filter or all(meta.get(key) == value for key, value in metadata_filter.items())
        ]

        if not filtered_ids:
            return []  # Return empty if no documents match the filter

        # Get embeddings for filtered IDs
        filtered_embeddings = np.array([
            self.index.reconstruct(self._get_index_for_id(doc_id))
            for doc_id in filtered_ids
        ]).astype("float32")

        # Create a temporary FAISS index for the filtered embeddings
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        # Search on the temporary index
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = temp_index.search(query_np, k)

        # Retrieve metadata for results
        results = [
            {
                "id": filtered_ids[idx],
                "distance": dist,
                "metadata": self.metadata[filtered_ids[idx]],
            }
            for dist, idx in zip(distances[0], indices[0]) if idx != -1
        ]

        return results
    
    def _get_index_for_id(self, doc_id):
        return list(self.metadata.keys()).index(doc_id)

class VoyageEmbedder:
    def __init__(self, voyage_client, batch_size=64):
        self.voyage_client = voyage_client
        self.batch_size = batch_size

    def embed_query(self, query):
        embedding_obj = self.voyage_client.multimodal_embed(
            inputs=[[query]],
            model="voyage-multimodal-3",
            input_type="query"
        )
        return embedding_obj.embeddings[0]


class VoyagePipeline(ABC):

    def __init__(self, config, task, index_file, metadata_file):
        self.config = config
        self.task = task
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.embedder = VoyageEmbedder(config.vo)
        self.faiss_db = FaissWithMetadata(index_file=index_file, metadata_file=metadata_file)

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

    @abstractmethod
    async def process_query(self, data,idx):
        pass

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

        with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/voyage/voyage_qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)

        logging.info("Processing completed and results saved.")
