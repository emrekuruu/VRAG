import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.colpali_pipeline import ColpaliPipeline

class TableVQAPipeline(ColpaliPipeline):

    def prepare_dataset(self):
        def process_qa_id(qa_id):
            splitted = qa_id.split(".")[0]
            return (
                splitted.split("_")[0]
                + "/"
                + splitted.split("_")[1]
                + "/"
                + splitted.split("_")[2]
                + "_"
                + splitted.split("_")[3]
                + ".pdf"
            )

        data = (
            load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"]
            .to_pandas()[["qa_id", "question", "gt"]]
        )
        data.qa_id = data.qa_id.apply(process_qa_id)
        data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
        data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
        data = data.rename(columns={"qa_id": "id"})

        data = data.iloc[:2]
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
