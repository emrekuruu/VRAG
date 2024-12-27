import json
import os
import logging
import pandas as pd
from datasets import load_dataset
from Generation.generation import hybrid
import asyncio
import math 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
model_type = "google"

with open(f".keys/openai_api_key.txt", "r") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()

class HybridPipeline:
    def __init__(self, task, max_concurrent_tasks=16):
        self.task = task
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        log_dir = f".logs/{self.task}"
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, "hybrid_generation.log"),
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def prepare_data(self, task):
        try:
            if task == "FinanceBench":
                data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
                data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
                data["evidence"] = data["evidence"].apply(lambda x: x[0]["evidence_text"])
                return data
            
            elif task == "FinQA":
                dataset = load_dataset("ibm/finqa", trust_remote_code=True)
                data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
                data.reset_index(drop=True, inplace=True)
                data = data[["id", "question", "answer", "gold_inds"]]
                data["Company"] = [row[0] for row in data.id.str.split("/")]
                data["Year"] = [row[1] for row in data.id.str.split("/")]
                data.id = data.id.map(lambda x: x.split("-")[0])
                return data

            elif task == "Table_VQA":
                def process_qa_id(qa_id):
                    splitted = qa_id.split(".")[0]
                    return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

                data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt", "text_html_table"]]
                data.qa_id = data.qa_id.apply(process_qa_id)
                data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
                data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
                data = data.rename(columns={"qa_id": "id", "gt": "answer", "text_html_table": "evidence"})
                return data

        except Exception as e:
            logging.error(f"Error preparing data for task {task}: {e}")
            raise

    def load_contexts(self):
        try:
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "r") as f:
                text_context = json.load(f)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/context.json"), "r") as f:
                image_context = json.load(f)

            return text_context, image_context

        except FileNotFoundError as e:
            logging.error(f"Context files not found: {e}")
            raise

    async def process_query(self, query_id, query, text_context, image_context):
        async with self.semaphore:
            try:
                query_id = str(query_id)
                text_data = text_context[query_id]
                image_data = image_context[query_id]
                answer = await hybrid(query=query, pages=image_data, chunks=text_data, model_type=model_type)
                logging.info(f"Generated answer for query {query_id}")
                return query_id, answer
            except Exception as e:
                logging.error(f"Error processing query {query_id}: {e}")
                return query_id, {"reasoning": "Error", "answer": "Error"}

    async def generate_answers(self, dataset, text_context, image_context, answers, batch_size=10):
        
        output_dir = os.path.join(parent_dir, f".results/{self.task}/generation/hybrid")

        for i in range(0, len(dataset["question"]), batch_size):
            logging.info(f"Processing batch {i // batch_size + 1} out of {math.ceil(len(dataset) / batch_size)}")
            tasks = []
            for query_id, query in dataset.iloc[i: i+batch_size]["question"].items():
                if str(query_id) in answers.keys():
                    continue
                else:
                    tasks.append(self.process_query(query_id, query, text_context, image_context))

            batch_results = await asyncio.gather(*tasks)

            for query_id, answer in batch_results:
                answers[query_id] = answer
            
            with open(os.path.join(output_dir, f"{model_type}_answers.json"), "w") as f:
                json.dump(answers, f, indent=4)

        return answers

    async def __call__(self):
        
        output_dir = os.path.join(parent_dir, f".results/{self.task}/generation/hybrid")
        os.makedirs(output_dir, exist_ok=True)
    
        try:
            dataset = self.prepare_data(self.task)
            text_context, image_context = self.load_contexts()

            if not os.path.exists(os.path.join(output_dir, f"{model_type}_answers.json")):
                answers = {}
            else:
                with open(os.path.join(output_dir, f"{model_type}_answers.json"), "r") as f:
                    answers = json.load(f)
                    answers = {k: v for k, v in answers.items() if v["reasoning"] != "Error"}

            answers = await self.generate_answers(dataset, text_context, image_context, answers)

            with open(os.path.join(output_dir, f"{model_type}_answers.json"), "w") as f:
                json.dump(answers, f, indent=4)

            logging.info("Pipeline finished successfully.")
            print("Finished")

        except Exception as e:
            logging.error(f"Error in pipeline execution: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    task = "FinanceBench"
    pipeline = HybridPipeline(task)
    asyncio.run(pipeline())
