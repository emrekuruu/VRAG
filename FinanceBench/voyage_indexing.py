from datasets import load_dataset
from BasePipelines.voyage_indexing import VoyageIndexing
import asyncio
from BasePipelines.config import Config
class FinanceBenchVoyageIndexing(VoyageIndexing):

    def prepare_metadata(self, file_key):
        metadata = {
            "Filename": file_key,
        }
        return metadata

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="finance-bench")
        pipeline = FinanceBenchVoyageIndexing(config=config, task="FinanceBench")
        await pipeline.run()

    asyncio.run(main())
