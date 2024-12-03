import os
import torch
import pandas as pd 
import asyncio
from datasets import load_dataset
import voyageai
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pickle
from math import ceil 
import logging
import json

# Configure logging
logging.basicConfig(
    filename="vanilla_retrieval.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(8)

# Load API key
with open("../keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()


with open("../keys/voyage_api_key.txt",  "r") as file:
    voyage_api_key = file.read().strip()


class Embedder:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size  

    def embed_document(self, text):
        embedding = vo.embed([text], model="voyage-3", input_type="document").embeddings[0]
        return embedding

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = vo.embed(batch, model="voyage-3", input_type="document").embeddings
            embeddings.extend([embedding for embedding in batch_embeddings])
        return embeddings
    
    def embed_query(self, query):
        embedding = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
        return embedding


vo = voyageai.Client(api_key=voyage_api_key)
embedder = Embedder()
persist_directory = ".chroma"


def prepare_dataset():
    # Load the dataset
    dataset = load_dataset("ibm/finqa", trust_remote_code=True)

    # Access the splits
    data = dataset['train'].to_pandas()
    validation_data = dataset['validation'].to_pandas()
    test_data = dataset['test'].to_pandas()

    data = pd.concat([data, validation_data, test_data])
    data.reset_index(drop=True, inplace=True)
    data.id = data.id.map(lambda x : x.split("-")[0])

    data = data[["id", "question", "answer", "gold_inds"]]
    data["Company"] = [row[0] for row in data.id.str.split("/")]
    data["Year"] = [row[1] for row in data.id.str.split("/")]

    return data

def read_pickle_file(filename):

    with open(filename, "rb") as f:
        documents = pickle.load(f)

    documents = { k: v for k, v in documents.items() if len(v) > 0 }
    chunks = [doc for key, docs in documents.items() for doc in docs if doc.page_content is not None]

    return chunks
    
def create_db(chunks):
    
    global vo
    global embedder

    if os.path.exists(persist_directory):
        # Load the existing ChromaDB
        chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedder)
        print("Loaded existing ChromaDB from .chroma")
    else:
        # Create ChromaDB and store the documents
        chroma_db = Chroma(
            embedding_function=embedder,
            persist_directory=persist_directory,
        )
        
        print("Created new ChromaDB and saved to .chroma")

        batch_size = 5000
        num_batches = ceil(len(chunks) / batch_size)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_docs = chunks[start_idx:end_idx]
            
            chroma_db.add_texts(
                texts=[doc.page_content for doc in batch_docs],
                metadatas=[doc.metadata for doc in batch_docs]
            )
            print(f"Batch {i+1} of {num_batches} added to ChromaDB.")
        
    return chroma_db

def sigmoid(x):
    return 1 / (1 + torch.exp(-torch.tensor(x)))

def rerank(query, documents, ids, top_k=5):
    scores = {}
    reranking = vo.rerank(query=query, documents=documents, model="rerank-2-lite", top_k=len(documents))

    for i, r in enumerate(reranking.results):
        normalized_score = sigmoid(r.relevance_score).item()
        scores[ids[i]] = normalized_score

    top_scorers = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return {id: score for id, score in top_scorers}

async def process_item(data, idx, chroma_db):
    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Initialize retriever
    retriever = chroma_db.as_retriever(
        search_kwargs={
            "k": 20,
            "filter": {
                "$and": [
                    {"Company": company},
                    {"Year": year},
                ]
            }
        }
    )

    max_retries = 5  # Maximum number of retries
    retry_delay = 5  # Initial delay between retries (in seconds)
    
    for attempt in range(max_retries):
        try:
            # Retrieve documents
            retrieved_docs = await asyncio.to_thread(retriever.invoke, query)
            
            logging.info(f"Retrieved documents {retrieved_docs} for index {idx} on attempt {attempt + 1}")

            # Rerank the retrieved documents
            retrieved = await asyncio.to_thread(
                rerank,
                query,
                [doc.page_content for doc in retrieved_docs],
                [doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] for doc in retrieved_docs]
            )
            
            # Log and return the result if successful
            logging.info(f"Retrieved context for index {idx} on attempt {attempt + 1}")
            return idx, retrieved

        except Exception as e:
            logging.error(f"Error processing index {idx} on attempt {attempt + 1}: {e}")

            # Check if it's a rate limit error and retry
            if "limit" in str(e).lower():
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                # For non-rate-limit errors, fail immediately
                logging.error(f"Non-recoverable error at index {idx}: {e}")
                return idx, {}

    # If all retries fail, log and return an empty result
    logging.error(f"Failed to process index {idx} after {max_retries} attempts.")
    return idx, {}


async def process_all(data, chroma_db, qrels):

    # Create tasks for processing each item
    tasks = [process_item(data, idx, chroma_db) for idx in data.index if str(idx) not in qrels.keys()]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the qrels dictionary
    for query_id, retrieved_qrels in results_list:
        qrels[query_id] = retrieved_qrels

    return qrels
    
async def main():
    
    data = prepare_dataset()
    chunks = read_pickle_file("processed_documents.pkl")
    chroma_db = create_db(chunks)

    with open("results/vanilla_qrels.json", "r") as f:
        qrels = json.load(f)

    qrels = {k: v for k, v in qrels.items() if len(v) > 0}

    # Generate qrels
    qrels = await process_all(data, chroma_db, qrels)

    # Save qrels to a JSON file for later use
    with open("results/vanilla_qrels.json", "w") as f:
        json.dump(qrels, f, indent=4)

    print("Qrels saved to results/qrels.json")
    
if __name__ == "__main__":
    asyncio.run(main())