from BasePipelines import Config
from BasePipelines import ColpaliIndexing
import asyncio

class FinanceBenchColpaliIndexing(ColpaliIndexing):
    def prepare_metadata(self, file_key):
        return {"Filename": file_key}

if __name__ == "__main__":

    config = Config(bucket_name="finance-bench")
    indexing = FinanceBenchColpaliIndexing(config=config, task="FinanceBench", index="finance_bench")

    asyncio.run(indexing.run())
