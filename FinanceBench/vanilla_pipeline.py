import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.vanilla_pipeline import TextPipeline
import pickle 
import os 

current_dir = os.path.dirname(__file__)

class FinanceBenchPipeline(TextPipeline):

    def read_chunks(self):
        with open( os.path.join(current_dir,"processed_documents.pkl") , "rb")  as f:
            documents = pickle.load(f)

        documents = { k: v for k, v in documents.items() if len(v) > 0 }
        chunks = [doc for key, docs in documents.items() for doc in docs if doc.page_content is not None]
        return chunks

    def prepare_dataset(self):
        data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
        data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
        return data

    async def retrieve(self, idx, data, top_n):
        
        query = data.loc[idx, "question"]
        filename = data.loc[idx, "doc_name"] + ".pdf"

        retrieved = await asyncio.to_thread( self.chroma_db.similarity_search,
            query=query,
            k=10, 
            filter={
                "Filename": filename,
            }
        )

        ids = [doc.metadata["Company"] + "/" + doc.metadata["Year"] + "/" + doc.metadata["Filename"] for doc in retrieved] 
        documents = [doc.page_content for doc in retrieved]

        return query, ids, documents

if __name__ == "__main__":
    config = Config(bucket_name="finance-bench")
    pipeline = FinanceBenchPipeline(config=config, task="FinanceBench",  persist_directory=os.path.join(current_dir, ".chroma"))
    asyncio.run(pipeline())
