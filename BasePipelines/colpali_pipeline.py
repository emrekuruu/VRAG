import asyncio
import logging
import json
from abc import ABC, abstractmethod
from byaldi import RAGMultiModalModel
from Generation.generation import image_based, evaluate_faithfulness
import os 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

class ColpaliPipeline(ABC):
    def __init__(self, config, task, index, device="mps"):
        self.config = config
        self.task = task
        self.index = index
        self.device = device

        logging.basicConfig(
            filename=f".logs/{self.task}-colpali_retrieval.log",
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
                logging.info(f"Done with query {idx}")

                sorted_retrieved = dict(sorted(retrieved.items(), key=lambda item: item[1]["score"], reverse=True))

                qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
                context = {k: v["base64"] for k, v in sorted_retrieved.items()}

                answer = await image_based(query, context.values())
                faithfullnes = await evaluate_faithfulness(query, answer, context.values(), type="image")

                logging.info(f"Done with query {idx}")

            except Exception as e:
                logging.warning(f"Error processing query {idx}: {e}")
                qrels = {}
                answer = ""
                faithfullnes = {"reasoning" : "Error", "score": 0.0}

            return idx, qrels, answer, faithfullnes

    async def process_all(self, data):
        qrels = {}
        answers = {}
        faithfullness = {}

        tasks = [
            self.process_item(data, idx)
            for idx in data.index
        ]

        results_list = await asyncio.gather(*tasks)

        for query_id, retrieved_qrels, answer, faithfullness_score in results_list:
            qrels[query_id] = retrieved_qrels
            answers[query_id] = answer
            faithfullness[query_id] = faithfullness_score

        return qrels, answers, faithfullness

    async def __call__(self):
        data = self.prepare_dataset()

        qrels, answers, faithfullness = await self.process_all(data)

        with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)

        # with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/answers.json"), "w") as f:
        #     json.dump(answers, f, indent=4)

        with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/faithfullness.json"), "w") as f:
            json.dump(faithfullness, f, indent=4)

        print("Finished")

