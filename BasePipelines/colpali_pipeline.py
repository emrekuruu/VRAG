import asyncio
import logging
import json
from abc import ABC, abstractmethod
from byaldi import RAGMultiModalModel
from Generation.generation import image_based
import os 
import pickle 
import math 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
model_type = "google"

class ColpaliPipeline(ABC):
    def __init__(self, config, task, index, device="mps"):
        self.config = config
        self.task = task
        self.index = index
        self.device = device

        logging.basicConfig(
            filename=f".logs/{self.task}/colpali_retrieval.log",
            filemode="w",  
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.aws_semaphore = asyncio.Semaphore(10)
        self.qrel_semaphore = asyncio.Semaphore(16)
        self.RAG = RAGMultiModalModel.from_index(
            index_path= os.path.join( parent_dir, f"{task}/.byaldi/{self.index}"), device=self.device
        )

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    async def retrieve(self, idx, data, top_k):
        pass

    async def process_item(self, data, idx, top_k=5):
        async with self.qrel_semaphore:

            try:
                query, retrieved = await self.retrieve(idx, data, top_k)
                sorted_retrieved = dict(sorted(retrieved.items(), key=lambda item: item[1]["score"], reverse=True))

                qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
                context = {k: v["base64"] for k, v in sorted_retrieved.items()}
                answer = await image_based(query, context.values(), model_type=model_type)

                logging.info(f"Done with query {idx}")

            except Exception as e:
                logging.warning(f"Error processing query {idx}: {e}")
                qrels = {}
                answer = ""

            return idx, qrels, answer, list(context.values())

    async def process_all(self, qrels, answers, context, data, batch_size=10):
        results = []

        for i in range(0, len(data), batch_size):
            tasks = []
            logging.info(f"Processing batch {i // batch_size + 1} out of {math.ceil(len(data) / batch_size)}")

            for j in range(i, i + batch_size):
                idx = data.index[j] if j < len(data) else None
                if str(idx) in answers.keys() or j >= len(data):
                    continue
                else:
                    tasks.append(self.process_item(data, idx))

            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            for idx, qrel, answer, query_context in results:
                qrels[idx] = qrel
                answers[idx] = answer
                context[idx] = query_context

            with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json"), "w") as f:
                json.dump(qrels, f, indent=4)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/{model_type}_answers.json"), "w") as f:
                json.dump(answers, f, indent=4)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/context.json"), "w") as f:
                json.dump(context, f, indent=4)

        return qrels, answers, context
    

    async def __call__(self):
        data = self.prepare_dataset()

        if not os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/image/{model_type}_answers.json")):
            qrels = {}
            answers = {}
            context = {}
        else:
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/{model_type}_answers.json"), "r") as f:
                answers = json.load(f)
            
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/context.json"), "r") as f:    
                context = json.load(f)

            qrels = {}

        qrels, answers, context = await self.process_all(qrels=qrels, context=context, answers=answers, data=data)

        with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)

        with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/{model_type}_answers.json"), "w") as f:
            json.dump(answers, f, indent=4)

        with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/context.json"), "w") as f:
            json.dump(context, f, indent=4)

        print("Finished")

