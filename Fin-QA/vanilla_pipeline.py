import os
import torch
import cohere
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
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
import math
import time 

# Configure logging
logging.basicConfig(
    filename="vanilla_retrieval.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(8)

# Load API key
with open("../.keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

with open("../.keys/voyage_api_key.txt",  "r") as file:
    voyage_api_key = file.read().strip()

with open("../.keys/cohere_api_key.txt",  "r") as file:
    cohere_api_key = file.read().strip()

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
    dataset = load_dataset("ibm/finqa", trust_remote_code=True)
    data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
    data.reset_index(drop=True, inplace=True)
    data = data[["id", "question", "answer", "gold_inds"]]
    data["Company"] = [row[0] for row in data.id.str.split("/")]
    data["Year"] = [row[1] for row in data.id.str.split("/")]
    data.id = data.id.map(lambda x: x.split("-")[0])
    return data 


def read_pickle_file(filename):

    with open(filename, "rb") as f:
        chunks = pickle.load(f)

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
    return 1 / (1 + math.exp(-x))

def rerank(query, documents, ids, reranker, top_k=5,):

    scores = {}

    if reranker == "cohere":

        co = cohere.Client(api_key=cohere_api_key)

        rerank_response = co.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model='rerank-v3.5'
        )

        for result in rerank_response.results:
            scores[ids[result.index]] = result.relevance_score

    elif reranker == "colbert":

        # Build a temporary index from documents
        temp_index = VectorStoreIndex.from_documents(
            [Document(text=doc, metadata={"id": doc_id}) for doc, doc_id in zip(documents, ids)]
        )

        # Initialize ColBERTv2 reranker
        colbert_reranker = ColbertRerank(
            top_n=top_k,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )

        # Create a query engine with ColBERT as a postprocessor
        query_engine = temp_index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[colbert_reranker],
        )

        # Query and retrieve results
        response = query_engine.query(query)

        # Extract scores from response
        for node in response.source_nodes:
            scores[node.metadata["id"]] = node.score

        # Sort and select top_k documents
        top_scorers = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        scores = {id: score for id, score in top_scorers}

    else:

        reranking = vo.rerank(query=query, documents=documents, model="rerank-2", top_k=len(documents))

        for i, r in enumerate(reranking.results):
            normalized_score = sigmoid(r.relevance_score)
            scores[ids[i]] = normalized_score

        top_scorers = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        scores = {id: score for id, score in top_scorers}

    return scores


async def process_item(data, idx, chroma_db, top_k, reranker):

    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    start_time = time.time()

    retrieved = await asyncio.to_thread(
        chroma_db.similarity_search_with_score,
        query=query,
        k=top_k, 
        filter={
                "$and": [
                    {"Company": company},
                    {"Year": year},
                    ]
        }
    )
     
    if reranker is None:
        
        qrels = {
            doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] : 1 / ( 1 + score)
            for doc, score in retrieved
            }
        
        query_latency = time.time() - start_time

        return idx, qrels, query_latency
    
    else:

        max_retries = 25 
        retry_delay = 5  
        
        for attempt in range(max_retries):
            try:
                # Rerank the retrieved documents
                retrieved = await asyncio.to_thread(
                    rerank,
                    query,
                    [doc.page_content for doc, score in retrieved],
                    [doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] for doc, score in retrieved],
                    reranker
                )
                
                # Log and return the result if successful
                logging.info(f"Retrieved context for index {idx} on attempt {attempt + 1}")

                query_latency = time.time() - start_time
                return idx, retrieved, query_latency

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
                    return idx, {}, -1

        logging.error(f"Failed to process index {idx} after {max_retries} attempts.")
        return idx, {}, -1


async def process_all(data, chroma_db, qrels, latency, top_k, reranker):

    # Create tasks for processing each item
    tasks = [process_item(data, idx, chroma_db, top_k, reranker) for idx in data.index if str(idx) not in qrels.keys()]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the qrels dictionary
    for query_id, retrieved_qrels, query_latency in results_list:
        qrels[query_id] = retrieved_qrels
        latency[query_id] = query_latency

    return qrels
    
async def main():
    data = prepare_dataset()
    chunks = read_pickle_file("chunks.pkl")
    chroma_db = create_db(chunks)
    
    rerankers = ["cohere", "reranker2"]
    top_ks = [5, 10, 20]
    
    for reranker in rerankers:
        for top_k in top_ks:
            qrels = {}
            latency = {}

            # Generate qrels
            qrels = await process_all(data, chroma_db, qrels, latency, top_k, reranker)

            # Save qrels to a JSON file for later use
            filename = f"results/text/vanilla_qrels_{reranker}_{top_k}.json"

            latency_filename = f"results/latency/vanilla_query_latency_{reranker}_{top_k}.json"

            with open(filename, "w") as f:
                json.dump(qrels, f, indent=4)

            with open(latency_filename, "w") as f:
                json.dump(latency, f, indent=4)
            
            print(f"Qrels saved to {filename}")

    # Done with ReRankers
    qrels = {}
    latency = {}
            
    qrels = await process_all(data, chroma_db, qrels, latency, 5, None)

    # Save qrels to a JSON file for later use
    filename = f"results/text/vanilla_qrels_retrieve_only.json"
    latency_filename = f"results/latency/vanilla_query_latency_retrieve_only.json"

    with open(filename, "w") as f:
        json.dump(qrels, f, indent=4)

    with open(latency_filename, "w") as f:
        json.dump(latency, f, indent=4)
    
    print(f"Qrels saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())