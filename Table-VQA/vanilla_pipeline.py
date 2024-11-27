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
import gzip


# Configure logging
logging.basicConfig(
    filename="vanilla_retrieval.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(16)

# Load API key
with open("../keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()


with open("../keys/voyage_api_key.txt",  "r") as file:
    voyage_api_key = file.read().strip()


class Embedder:
    def __init__(self, batch_size=128):
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

    def process_qa_id(qa_id):
        splitted = qa_id.split(".")[0]
        return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

    data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt"]]
    data.qa_id = data.qa_id.apply(process_qa_id)
    data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
    data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
    data = data.rename(columns={"qa_id": "id"})
    return data

def read_pickle_file(filename):
    """Read all objects from a pickle file that contains multiple appended objects."""
    data = []
    try:
        with open(filename, "rb") as file:
            while True:
                try:
                    # Load each pickled object
                    batch = pickle.load(file)
                    data.extend(batch)  # Add batch data to the result list
                except EOFError:
                    # End of file reached
                    break
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
    return data
    
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

def rerank(query, documents, ids, top_k=1):
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
    retriever = chroma_db.as_retriever(search_kwargs={"k": 20, "filter": {"$and": [{"Company": company}, {"Year": year}]}})
    
    # Retrieve and rerank
    retrieved_docs = await asyncio.to_thread( retriever.invoke, query)
    retrieved = rerank(query, [doc.page_content for doc in retrieved_docs], [doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] for doc in retrieved_docs])

    retrieved_context = list(retrieved.keys())[0]
    logging.info(f"Retrieved context for index {idx}")

    return idx, retrieved_context


async def process_all(data, chroma_db):

    # Initialize the results DataFrame
    results = pd.DataFrame(columns=["Retrieved Context"], index=data.index)

    # Create tasks for processing each item
    tasks = [process_item(data, idx, chroma_db) for idx in data.index]

    # Gather results asynchronously
    results_list = await asyncio.gather(*tasks)

    # Populate the results DataFrame
    for idx, retrieved_context in results_list:
        results.loc[idx, "Retrieved Context"] = retrieved_context

    return results
    

async def main():
    data = prepare_dataset()
    docs = (read_pickle_file("processed_documents.pkl"))
    chunks = [chunk for file, chunks in docs for chunk in chunks if chunk.page_content.strip() != ""]
    chroma_db = create_db(chunks)
    results = await process_all(data, chroma_db)
    results.to_csv("results/vanilla.csv", index=True)
    
if __name__ == "__main__":
    asyncio.run(main())
