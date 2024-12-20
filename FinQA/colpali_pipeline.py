import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.colpali_pipeline import ColpaliPipeline
import pandas as pd 

class TableVQAPipeline(ColpaliPipeline):


    def prepare_dataset(self):
        dataset = load_dataset("ibm/finqa", trust_remote_code=True)
        data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
        data.reset_index(drop=True, inplace=True)
        data = data[["id", "question", "answer", "gold_inds"]]
        data["Company"] = [row[0] for row in data.id.str.split("/")]
        data["Year"] = [row[1] for row in data.id.str.split("/")]
        data.id = data.id.map(lambda x: x.split("-")[0])
        return data 

    async def retrieve(self, idx, data, top_n):
        query = data.iloc[idx]["question"]
        company = data.iloc[idx]["Company"]
        year = data.iloc[idx]["Year"]

        retrieved = await asyncio.to_thread(
            self.RAG.search, query, k=top_n, filter_metadata={"Company_Year": f"{company}_{year}"}
        )

        results = {}
        for doc in retrieved:
            file_key = doc.metadata["Filename"]
            base64_images = await self.config.fetch_file_as_base64_images(
                file_key=f"{company}/{year}/{doc.metadata["Filename"]}",
                filename=file_key,
                semaphore=self.aws_semaphore,
            )
            results[file_key] = {"score": doc.score, "base64": base64_images}

        return query, results

if __name__ == "__main__":
    config = Config(bucket_name="table-vqa")
    pipeline = TableVQAPipeline(config=config, task="Table_VQA", index="table_vqa")
    asyncio.run(pipeline())
