from datasets import load_dataset
from BasePipelines.voyage_pipeline import VoyagePipeline
import asyncio
from BasePipelines.config import Config
import os 

current_dir = os.path.dirname(__file__)
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
            query = data.loc[idx, "question"]
            company = data.loc[idx, "Company"]
            year = data.loc[idx, "Year"]

            query_embedding = self.embedder.embed_query(query)

            results = asyncio.to_thread( self.faiss_db.search, query_embedding, k=5, metadata_filter={"Company": company, "Year": year})

            qrels = {
                idx: {
                    (result["metadata"]["Company"] + "/" + result["metadata"]["Year"] + "/" + result["metadata"]["Filename"]): 1 / (1 + float(result["distance"]))
                    for result in results
                }
            }

            print("Done with query ", idx)

        except Exception as e:
            qrels = {idx: {}}
            print(e)

        return qrels

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="table-vqa")
        pipeline = TableVQAVoyagePipeline(config=config, task="Table_VQA", index_file=os.path.join(current_dir,"faiss_index.bin"), metadata_file=os.path.join(current_dir, "metadata.json"))
        await pipeline()

    asyncio.run(main())
