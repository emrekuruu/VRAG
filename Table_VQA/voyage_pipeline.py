import os
import asyncio
import logging
import voyageai
import fitz
from PIL import Image
import json
import numpy as np
import faiss
from datasets import load_dataset

logging.basicConfig(
    filename="retrieval.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

semaphore = asyncio.Semaphore(16)

# AWS Configuration
key_folder = "../keys"

with open(f"{key_folder}/voyage_api_key.txt", "r") as file:
    voyage_api_key = file.read().strip()

vo = voyageai.AsyncClient(api_key=voyage_api_key)

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
        """Helper to find the index of a document ID."""
        return list(self.metadata.keys()).index(doc_id)

class Embedder:
    def __init__(self):
        pass

    async def embed_query(self, query):
        embedding_obj = await vo.multimodal_embed(
            inputs=[[query]],
            model="voyage-multimodal-3",
            input_type="query"
        )
        return embedding_obj.embeddings[0]

embedder = Embedder()

async def process_query(data, idx, faiss_db):
    try:
        query = data.loc[idx, "question"]
        company = data.loc[idx, "Company"]
        year = data.loc[idx, "Year"]

        # Embed the query
        query_embedding = await embedder.embed_query(query)

        # Perform search with metadata filtering
        results = faiss_db.search(query_embedding, k=5, metadata_filter={"Company": company, "Year": year})

        # Construct QRELs
        qrels = {
            idx: {
                (result["metadata"]["Company"] + "/" + result["metadata"]["Year"] + "/" + result["metadata"]["Filename"]): 1 / (1 + float(result["distance"]))
                for result in results
            }
        }

    except Exception as e:
        logging.error(f"Error processing query {idx}: {e}")
        qrels = {idx: {}}

    return qrels

async def process_queries(data, faiss_db):
    qrels = {}
    tasks = [process_query(data, idx, faiss_db) for idx in data.index]
    results_list = await asyncio.gather(*tasks)
    for result in results_list:
        qrels.update(result)
    return qrels

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

async def main():
    index_file = "faiss_index.bin"
    metadata_file = "metadata.json"

    faiss_db = FaissWithMetadata(index_file=index_file, metadata_file=metadata_file)
    data = prepare_dataset()
    qrels = await process_queries(data, faiss_db)

    with open("results/voyage_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/voyage_qrels.json")

if __name__ == "__main__":
    asyncio.run(main())
