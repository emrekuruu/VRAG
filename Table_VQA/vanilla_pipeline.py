import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.vanilla_pipeline import TextPipeline
import pickle 

class TableVQATextPipeline(TextPipeline):

    def read_chunks(self):
        with open("/Users/emrekuru/Developer/VRAG/Table_VQA/processed_documents.pkl", "rb") as f:
            documents = pickle.load(f)

        documents = { k: v for k, v in documents.items() if len(v) > 0 }
        chunks = [doc for key, docs in documents.items() for doc in docs if doc.page_content is not None]
        return chunks

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

    async def retrieve(self, idx, data, top_n):
        query = data.loc[idx, "question"]
        company = data.loc[idx, "Company"]
        year = data.loc[idx, "Year"]
        
        retrieved = await self.chroma_db.asimilarity_search(
            query=query,
            k=top_n, 
            filter={
                    "$and": [
                        {"Company": company},
                        {"Year": year},
                        ]
            }
        )

        ids = [doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] for doc in retrieved] 
        documents = [doc.page_content for doc in retrieved]

        return query, ids, documents

if __name__ == "__main__":
    config = Config(bucket_name="table-vqa")
    pipeline = TableVQATextPipeline(config=config, task="Table_VQA", persist_directory="/Users/emrekuru/Developer/VRAG/Table_VQA/.chroma")
    asyncio.run(pipeline())
