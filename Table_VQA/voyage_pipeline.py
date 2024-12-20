from datasets import load_dataset
from BasePipelines.voyage_pipeline import VoyagePipeline
import asyncio
from BasePipelines.config import Config
class TableVQAVoyagePipeline(VoyagePipeline):

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
        return data

    async def process_query(self, data, idx):
        try:
            query = data.loc[idx, 'question']
            filename = data.loc[idx, 'doc_name'] + ".pdf"

            query_embedding = await self.embedder.embed_query(query)

            results = await self.search(query_embedding, k=5, metadata_filter={"Filename": filename})

            qrels = {
                idx: {
                    result['id']: 1 / (1 + float(result['distance']))
                    for result in results
                }
            }

        except Exception as e:
            qrels = {idx: {}}

        return qrels

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="table-vqa")
        pipeline = TableVQAVoyagePipeline(config=config, task="Table_VQA")
        await pipeline()

    asyncio.run(main())
