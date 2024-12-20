from BasePipelines.vanilla_indexing import Chunking
from BasePipelines.config import Config
import asyncio

class FinQAChunking(Chunking):

    def prepare_metadata(self, chunk, file_key):
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]
    
        return {
           "category": chunk.category, "Company": company, "Year": year, "Filename": filename
        }
    
if __name__ == "__main__":

    config = Config(bucket_name="colpali-docs")
    chunking_processor = FinQAChunking(config=config, task="FinQA")
    asyncio.run(chunking_processor.run())