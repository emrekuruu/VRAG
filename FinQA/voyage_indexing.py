from datasets import load_dataset
from BasePipelines.vanilla_indexing import VoyageIndexing
import asyncio
from BasePipelines.config import Config

class FinQAVoyageIndexing(VoyageIndexing):

    def prepare_metadata(self, file_key):
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]
    
        metadata = {
            "Company": company,
            "Year": year,
            "Filename": filename,
        }

        return metadata

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="colpali-docs")
        pipeline = FinQAVoyageIndexing(config=config, task="FinQAA")
        await pipeline.run()

    asyncio.run(main())