import asyncio
import logging
import json
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
from Generation.generation import text_based
import math 
import os 
import pickle 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
model_type = "claude"

class Embedder:
    def __init__(self, vo, batch_size=64):
        self.batch_size = batch_size  
        self.vo = vo

    def embed_document(self, text):
        embedding = self.vo.embed([text], model="voyage-3", input_type="document").embeddings[0]
        return embedding

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.vo.embed(batch, model="voyage-3", input_type="document").embeddings
            embeddings.extend([embedding for embedding in batch_embeddings])
        return embeddings
    
    def embed_query(self, query):
        embedding = self.vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
        return embedding


class TextPipeline(ABC):

    def __init__(self, config, task, persist_directory=".chroma"):
        self.config = config
        self.task = task
        self.persist_directory = persist_directory
        self.embedder = Embedder(self.config.vo, batch_size=64)

        logging.basicConfig(
            filename=f".logs/{self.task}/text_retrieval.log",
            filemode="w",  
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.qrel_semaphore = asyncio.Semaphore(32)

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    def read_chunks(self):
        pass

    async def create_db(self, chunks, batch_size=500):

        if os.path.exists(self.persist_directory):
            self.chroma_db = Chroma(persist_directory=self.persist_directory,embedding_function=self.embedder)
            logging.info("Loaded existing ChromaDB.")
        else:
            self.chroma_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
            logging.info("Created a new ChromaDB.")

            num_batches = math.ceil(len(chunks) / batch_size)
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, len(chunks))
                batch = chunks[batch_start:batch_end]

                self.chroma_db.add_texts(
                    texts=[chunk.page_content for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch]
                )
                logging.info(f"Added batch {i + 1}/{num_batches} to ChromaDB.")

        return self.chroma_db
    
    @abstractmethod
    async def retrieve(self, idx, data, top_n):
        pass

    async def rerank(self, query, documents, ids, top_k=5):
        scores = {}

        rerank_response = await asyncio.to_thread(
            self.config.co.rerank,
            query=query,
            documents=documents,
            top_n=top_k,
            model='rerank-v3.5'
        )

        for result in rerank_response.results:
            scores[ids[result.index]] = {
                "score": result.relevance_score,
                "content": documents[result.index]
            }

        return scores
    
    async def process_item(self, data, idx, top_n = 10):  
        async with self.qrel_semaphore:
            query, ids, documents = await self.retrieve(idx, data, top_n)
            reranked = await self.rerank(query, documents, ids)
            sorted_retrieved = dict(sorted(reranked.items(), key=lambda item: item[1]["score"], reverse=True))

            qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
            context = {k: v["content"] for k, v in sorted_retrieved.items()}

            answer =  await text_based(query, context, model_type=model_type)
            logging.info(f"Done with query {idx}")
            return idx, qrels, answer, list(context.values())

    async def process_all(self, qrels, answers, context, data, batch_size=10, top_n=10):
        results = []

        for i in range(0, len(data), batch_size):
            tasks = []
            logging.info(f"Processing batch {i // batch_size + 1} out of {math.ceil(len(data) / batch_size)}")
            for j in range(i, i + batch_size):
                idx = data.index[j]
                if str(idx) in answers.keys() or j >= len(data):
                    continue
                else:
                    tasks.append(self.process_item(data, idx, top_n))

            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            for idx, qrel, answer, query_context in results:
                qrels[idx] = qrel
                answers[idx] = answer
                context[idx] = query_context

            with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/text/text_qrels.json"), "w") as f:
                json.dump(qrels, f, indent=4)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/{model_type}_answers.json"), "w") as f:
                json.dump(answers, f, indent=4)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "w") as f:
                json.dump(context, f, indent=4)

        return qrels, answers, context

    async def __call__(self):

        data = self.prepare_dataset()

        chunks = self.read_chunks()
        self.chroma_db = await self.create_db(chunks)

        if not os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/text/{model_type}_answers.json")):
            qrels = {}
            answers = {}
            context = {}
        else:
            with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/text/text_qrels.json"), "r") as f:
                qrels = json.load(f)
            
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/{model_type}_answers.json"), "r") as f:
                answers = json.load(f)
            
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "r") as f:    
                context = json.load(f)

        qrels, answers, context = await self.process_all(qrels=qrels, context=context, answers=answers, data=data)

        with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/text/text_qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)

        with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/{model_type}_answers.json"), "w") as f:
            json.dump(answers, f, indent=4)

        with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "w") as f:
            json.dump(context, f, indent=4)

        print("Finished")
