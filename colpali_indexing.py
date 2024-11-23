import os
import pandas as pd 
import asyncio
import os
import psutil

from datasets import load_dataset

from byaldi import RAGMultiModalModel
import time

# Run all tasks concurrently with a limit on concurrency
concurrency_limit = 4
semaphore = asyncio.Semaphore(concurrency_limit)

companies = None
RAG = None
data = None


# Define an async function to process a single page
async def process_page_for_index(company, year, page):
    async with semaphore:
        global RAG
        print(f"Memory usage: {psutil.virtual_memory().percent}% - Processing {company}/{year}/{page}")
        
        # Simulate asynchronous RAG indexing (replace with actual async-compatible call if possible)
        await asyncio.to_thread(
            RAG.add_to_index,
            input_item=f"docs/{company}/{year}/{page}",
            store_collection_with_index=True,
            metadata={"Company": company, "Year": year, "Page": page},
        )

# Define an async function to process all companies, years, and pages
async def process_all():
    global companies
    tasks = []

    for company in companies:
        years = os.listdir(f"docs/{company}/")
        years = ["2014"]

        for year in years:
            pages = os.listdir(f"docs/{company}/{year}/")

            # Create async tasks for each page
            for page in pages:
                tasks.append(asyncio.create_task(process_page_for_index(company, year, page)))

    await asyncio.gather(*tasks)
    
    return tasks

async def main():

    global companies, RAG, data, model

    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", device="mps")

    RAG.index(
        input_path="docs/temp/",
        index_name="finqa",
        overwrite=True,
    )

    # Define companies
    companies = ["AAL"]

    start_time = time.time()
    tasks = await process_all()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())