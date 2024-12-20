from BasePipelines.vanilla_indexing import Chunking
from BasePipelines.config import Config
import asyncio

class FinanceBenchChunking(Chunking):

    def prepare_metadata(self, chunk, file_key):
        return {
            "category": chunk.category, "Filename": file_key, "page_num": chunk.metadata.page_number
        }
    
if __name__ == "__main__":

    config = Config(bucket_name="finance-bench")
    chunking_processor = FinanceBenchChunking(config=config, task="FinanceBench")
    asyncio.run(chunking_processor.run())