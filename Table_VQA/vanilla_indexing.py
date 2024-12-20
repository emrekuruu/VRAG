from BasePipelines.vanilla_indexing import Chunking
from BasePipelines.config import Config
import asyncio

class TableVQAChunking(Chunking):

    def prepare_metadata(self, chunk, file_key):
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]
    
        return {
           "category": chunk.category, "Company": company, "Year": year, "Filename": filename
        }
    
if __name__ == "__main__":

    config = Config(bucket_name="table-vqa")
    chunking_processor = TableVQAChunking(config=config, task="Table_VQA")
    asyncio.run(chunking_processor.run())