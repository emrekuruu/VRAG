import asyncio
import logging
import os
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import pickle 

class Chunking(ABC):

    def __init__(self, config, task, temp_dir="/tmp/docs/temp", concurrency_limit=8):
        self.config = config
        self.temp_dir = temp_dir
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        logging.basicConfig(
            filename=f".logs{task}-chunking.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    @abstractmethod
    def prepare_metadata(self, chunk, file_key):
        """
        Abstract method to prepare metadata for a given chunk and file key.
        Subclasses must implement this method.
        """
        pass

    async def process_file(self, file_key):
        documents = {file_key: []}

        async with self.semaphore:
            local_file_path = os.path.join(self.temp_dir, os.path.basename(file_key))

            try:
                # Download file from S3
                await asyncio.to_thread(
                    self.config.s3_client.download_file,
                    self.config.bucket_name,
                    file_key,
                    local_file_path,
                )

                try:
                    # Partition PDF with table inference
                    elements = await asyncio.to_thread(partition_pdf, filename=local_file_path, strategy="hi_res", infer_table_structure=True)
                except Exception as e:
                    logging.warning(f"Table inference failed for {file_key}: {e}, retrying without table inference.")
                    # Retry partitioning without table inference
                    elements = await asyncio.to_thread(partition_pdf, filename=local_file_path, strategy="hi_res")

                # Chunk elements
                chunks = chunk_by_title(
                    elements,
                    combine_text_under_n_chars=1500,
                    max_characters=int(1e6),
                )

                for chunk in chunks:
                    metadata = self.prepare_metadata(chunk, file_key)
                    documents[file_key].append(
                        Document(
                            page_content=chunk.text,
                            metadata={**metadata, "category": chunk.category},
                        )
                    )

            except Exception as e:
                logging.error(f"Failed to process file {file_key}: {e}")

            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)

        return documents

    async def process_all_files(self, prefix=""):
        file_keys = self.config.list_s3_files(prefix=prefix)
        print(file_keys)
        tasks = [self.process_file(file_key) for file_key in file_keys]
        results = await asyncio.gather(*tasks)

        documents = {}
        for result in results:
            if result:
                documents.update(result)

        logging.info("All files processed successfully.")
        return documents

    async def run(self, prefix=""):

        os.makedirs(self.temp_dir, exist_ok=True)
        documents = await self.process_all_files(prefix=prefix)

        with open("processed_documents.pkl", "wb") as f:
            pickle.dump(documents, f)

        logging.info("Chunking process completed and results saved.")
