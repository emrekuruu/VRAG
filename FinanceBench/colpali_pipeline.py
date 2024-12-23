import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.colpali_pipeline import ColpaliPipeline

class FinanceBenchColpaliPipeline(ColpaliPipeline):

    def prepare_dataset(self):
        data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
        data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
        return data 

    async def retrieve(self, idx, data, top_n):
        query = data.iloc[idx]["question"]
        query = data.loc[idx, "question"]
        filename = data.loc[idx]["doc_name"] + ".pdf"

        retrieved = await asyncio.to_thread(
            self.RAG.search, query, k=top_n, filter_metadata={"Filename" : filename}
        )

        results = {}
        for doc in retrieved:
            file_key = doc.metadata["Filename"]
            base64_images = await self.config.fetch_file_as_base64_images(
                file_key=doc.metadata["Filename"],
                filename=file_key,
                semaphore=self.aws_semaphore,
                page = doc.page_num  
            )
            file_key = file_key + "_page_" + str(doc.page_num - 1)
            results[file_key] = {"score": doc.score, "base64": base64_images}

        return query, results

if __name__ == "__main__":
    config = Config(bucket_name="finance-bench")
    pipeline = FinanceBenchColpaliPipeline(config=config, task="FinanceBench", index="finance_bench")
    asyncio.run(pipeline())
