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

# Load API key
with open("../keys/openai_api_key.txt", "r") as file:
    openai_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

# Load input data
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
    """Process all file keys and save results in batches to the same file asynchronously."""
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

    # Prepare tasks for processing
    tasks = []
    for file_key in document_texts.keys():
        task = asyncio.create_task(process_item(file_key, splitter))
        tasks.append(task)

    # Process results in batches of 1000
    batch_size = 1000
    total_tasks = len(tasks)
    output_file = "processed_documents.pkl"

    for i in range(0, total_tasks, batch_size):
        # Get a batch of tasks
        batch_tasks = tasks[i:i + batch_size]

        # Gather results for the current batch
        batch_results = await asyncio.gather(*batch_tasks)
        batch_results = [result for result in batch_results if result is not None]

        # Append the current batch to the pickle file
        try:
            with open(output_file, "ab") as file:
                pickle.dump(batch_results, file)
            print(f"Added batch {i // batch_size + 1} to {output_file}")
        except Exception as e:
            logging.error(f"Error saving batch {i // batch_size + 1}: {e}")
            raise


async def main():
    """Main function to process documents and initialize the pickle file."""
    output_file = "processed_documents.pkl"

    # Initialize pickle file
    with open(output_file, "wb") as file:
        pass  # Create or clear the file to ensure it's ready for writing

    # Start processing
    logging.info("Starting document processing...")
    await process_all()
    logging.info(f"Document processing completed. Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
