import asyncio
from datasets import load_dataset
from BasePipelines.config import Config
from BasePipelines.vanilla_pipeline import TextPipeline
import pickle 
import os 
import pandas as pd 

current_dir = os.path.dirname(__file__)

class FinQATextPipeline(TextPipeline):

    def read_chunks(self):
        with open( os.path.join(current_dir,"chunks.pkl") , "rb")  as f:
            chunks = pickle.load(f)
        return chunks

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
    config = Config(bucket_name="colpali-docs")
    pipeline = FinQATextPipeline(config=config, task="FinQA", persist_directory=os.path.join(current_dir, ".chroma"))
    asyncio.run(pipeline())
