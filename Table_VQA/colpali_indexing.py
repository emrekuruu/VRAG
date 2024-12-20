from BasePipelines import Config
from BasePipelines import ColpaliIndexing
import asyncio

class TableVQAColpaliIndexing(ColpaliIndexing):
    def prepare_metadata(self, file_key):
        parts = file_key.split('/')
        company, year, filename = parts[0], parts[1], parts[2]
        return {"Company": company, "Year": year, "Filename": filename}

if __name__ == "__main__":

    config = Config(bucket_name="table-vqa")
    indexing = TableVQAColpaliIndexing(config=config, task="Table_VQA", index="table_vqa")

    asyncio.run(indexing.run())
