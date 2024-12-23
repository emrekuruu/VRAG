import asyncio
import logging
import os
from abc import ABC, abstractmethod
from byaldi import RAGMultiModalModel

class ColpaliIndexing(ABC):

    def __init__(self, config, task,  index, temp_dir="/tmp/docs/temp", concurrency_limit=8, device="mps"):
        self.config = config
        self.temp_dir = temp_dir
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.RAG =  RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", device=device)
        self.index = index

        logging.basicConfig(
            filename=f".logs/{task}-colpali-indexing.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    @abstractmethod
    def prepare_metadata(self, file_key):
        pass

    async def initialize_rag(self):

        os.makedirs(self.temp_dir, exist_ok=True)
        self.config.download_s3_folder("temp/", self.temp_dir)

        await asyncio.to_thread(
            self.RAG.index,
            input_path=self.temp_dir,
            index_name=self.index,
            overwrite=True,
        )

        logging.info("Index initialized successfully.")

    async def process_file(self, file_key):

        async with self.semaphore:
            metadata = self.prepare_metadata(file_key)
            local_file_path = os.path.join(self.temp_dir, os.path.basename(file_key))

            await asyncio.to_thread(
                self.config.s3_client.download_file,
                self.config.bucket_name,
                file_key,
                local_file_path,
            )

            await asyncio.to_thread(
                self.RAG.add_to_index,
                input_item=local_file_path,
                store_collection_with_index=True,
                metadata=metadata,
            )

            os.remove(local_file_path)
            logging.info(f"Processed file: {file_key}")

    async def process_all_files(self, prefix=""):

        file_keys = self.config.list_s3_files(prefix=prefix)
        tasks = [self.process_file(file_key) for file_key in file_keys]
        await asyncio.gather(*tasks)
        logging.info("All files processed successfully.")

    async def upload_index(self, s3_prefix="byaldi"):

        index_dir_path = os.path.join(os.getcwd(), ".byaldi")
        if os.path.exists(index_dir_path):
            self.config.upload_directory_to_s3(index_dir_path, self.config.bucket_name, s3_prefix)
            logging.info("Index directory uploaded successfully.")
        else:
            logging.error(f"Index directory not found: {index_dir_path}")

    async def run(self):

        await self.initialize_rag()
        await self.process_all_files()
        await self.upload_index()
        logging.info("Indexing process completed.")

