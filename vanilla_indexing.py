import os
import asyncio
import json
import logging
import pickle
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    filename="chunking.log",  # Log file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)

semaphore = asyncio.Semaphore(16)

with open("keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

with open("document_texts.json", "r") as file:
    document_texts = json.load(file)


async def process_item(file_key, splitter):
    """Process a single file key, splitting its content and returning documents."""
    global document_texts
    parts = file_key.split('/')
    company, year, filename = parts[0], parts[1], parts[2]

    try:
        text = document_texts[file_key]
    except KeyError:
        try:
            text = document_texts["docs/" + file_key]
        except KeyError:
            logging.error(f"Text not found for {file_key}")
            return None

    chunks = await asyncio.to_thread(splitter.split_text, text)

    docs = [
        Document(page_content=chunk, metadata={"Company": company, "Year": year, "Filename": filename})
        for chunk in chunks
    ]

    logging.info(f"Processed {file_key}")
    return file_key, docs


async def process_all():
    """Process all file keys and save results asynchronously."""
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Configure the SemanticChunker
    splitter = SemanticChunker(
        embeddings=embeddings,
        buffer_size=3,  # Number of sentences to group together
        add_start_index=True,  # Include start index in metadata
        breakpoint_threshold_type="percentile",  # Method to determine breakpoints
        breakpoint_threshold_amount=0.8,  # Threshold for splitting
        number_of_chunks=None,  # Let the chunker decide the number of chunks
        sentence_split_regex=r'(?<=\.)\s+'  # Split sentences based on periods
    )

    tasks = []

    async with semaphore:
        for file_key in document_texts.keys():
            # Wrap each task creation in the semaphore
            task = asyncio.create_task(process_item(file_key, splitter))
            tasks.append(task)

    # Use asyncio.gather to wait for all tasks to finish
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

    
async def main():
    """Main function to process documents and save them as a pickle file."""
    # Process all documents
    logging.info("Starting document processing...")
    processed_data = await process_all()

    # Save the results
    output_dict = {file_key: docs for file_key, docs in processed_data}
    output_file = "processed_documents.pkl"

    with open(output_file, "wb") as file:
        pickle.dump(output_dict, file)

    logging.info(f"Processed data saved to {output_file}")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())