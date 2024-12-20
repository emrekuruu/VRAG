import asyncio
import logging
import json
from abc import ABC, abstractmethod
from byaldi import RAGMultiModalModel
from Generation.generation import image_based

class ColpaliPipeline(ABC):
    def __init__(self, config, task, index, device="mps"):
        self.config = config
        self.task = task
        self.index = index
        self.device = device

        logging.basicConfig(
            filename=f".logs{self.task}-colpali_retrieval.log",
            filemode="w",  
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.aws_semaphore = asyncio.Semaphore(10)
        self.qrel_semaphore = asyncio.Semaphore(16)
        self.RAG = RAGMultiModalModel.from_index(
            index_path=f"/Users/emrekuru/Developer/VRAG/{self.task}/.byaldi/{self.index}", device=self.device
        )

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    async def retrieve(self, idx, data, top_k):
        pass

    async def process_item(self, data, idx, top_k=5):
        async with self.qrel_semaphore:
            query, retrieved = await self.retrieve(idx, data, top_k)

            sorted_retrieved = dict(sorted(retrieved.items(), key=lambda item: item[1]["score"], reverse=True))

            qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
            context = {k: v["base64"] for k, v in sorted_retrieved.items()}

            answer = await image_based(query, context.values())

            logging.info(f"Done with query {idx}")

            return idx, qrels, answer, context

    async def process_all(self, data):
        qrels = {}
        answers = {}
        context = {}

        tasks = [
            self.process_item(data, idx)
            for idx in data.index
        ]

        results_list = await asyncio.gather(*tasks)

        for query_id, retrieved_qrels, answer, query_context in results_list:
            qrels[query_id] = retrieved_qrels
            answers[query_id] = answer
            context[query_id] = query_context

        return qrels, answers, context

    async def __call__(self):
        data = self.prepare_dataset()

        qrels, answers, context = await self.process_all(data)

        with open(f"/Users/emrekuru/Developer/VRAG/.results/{self.task}/retrieval/colpali/colpali_qrels.json", "w") as f:
            json.dump(qrels, f, indent=4)

        with open(f"/Users/emrekuru/Developer/VRAG/.results/{self.task}/generation/image_answers.json", "w") as f:
            json.dump(answers, f, indent=4)

        with open(f"/Users/emrekuru/Developer/VRAG/.results/{self.task}/generation/image_context.json", "w") as f:
            json.dump(context, f, indent=4)

        print("Finished")

