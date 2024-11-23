import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd 
import asyncio
import os
import psutil
import nest_asyncio
import warnings

from datasets import load_dataset

from byaldi import RAGMultiModalModel

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Run all tasks concurrently with a limit on concurrency
concurrency_limit = 4
semaphore = asyncio.Semaphore(concurrency_limit)

companies = None
RAG = None
data = None

# Define the function to process a single item
def process_item(idx):
    global model
    query = data.loc[idx, "question"]
    company = data.loc[idx, "Company"]
    year = data.loc[idx, "Year"]

    # Perform retrieval
    retrieved = RAG.search(query, k=1, filter_metadata={"Company": company, "Year": year})

    # Populate the results row
    retrieved_context = f"{company}/{year}/{retrieved[0].metadata['Page']}"
    generated_answer = model.invoke([image_prompt(retrieved[0]["base64"], query)]).content

    return idx, retrieved_context, generated_answer

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

def image_prompt(image, question):

    query = f"""
    Answer the following query based solely on the provided image,  Give a short answer, 2-3 words at most. Then explain the steps you took to arrive at your answer.

    Query: {question}
    """

    message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/pdf;base64,{image}"},
        },
    ],
    )

    return message

async def main():
    global companies, RAG, data, model
    with open('keys/hf_key.txt', 'r') as file:
        hf_key = file.read().strip()

    with open("keys/openai_api_key.txt", "r") as file:
        openai_key = file.read().strip()

    os.environ["HF_TOKEN"] = hf_key
    os.environ["OPENAI_API_KEY"] = openai_key

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings('ignore')

    # Load the dataset
    dataset = load_dataset("ibm/finqa", trust_remote_code=True)

    # Access the splits
    data = dataset['train'].to_pandas()
    validation_data = dataset['validation'].to_pandas()
    test_data = dataset['test'].to_pandas()

    data = pd.concat([data, validation_data, test_data])
    data.reset_index(drop=True, inplace=True)
    data = data[["id", "question", "answer", "gold_inds"]]

    data["Company"] = [row[0] for row in data.id.str.split("/")]
    data["Year"] = [row[1] for row in data.id.str.split("/")]

    unique_companies = set(data.Company.unique())

    needed_years = {}

    for company in unique_companies:
        needed_years[company] = list(data[data.Company == company].Year.unique())

    file_count = 0

    for company in needed_years.keys():
        for year in needed_years[company]:
            try:
                file_count += len(os.listdir(f"docs/{company}/{year}/"))
            except:
                print(f"docs/{company}/{year}/")
                
    data = data[(data.Company == "AAL" )& (data.Year == "2014")]

    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", device="mps")

    RAG.index(
        input_path="docs/temp/",
        index_name="finqa",
        overwrite=True,
    )

    # Apply nest_asyncio to enable nested event loops in Jupyter
    nest_asyncio.apply()

    # Define companies
    companies = ["AAL"]

    tasks = await process_all()

    model = ChatOpenAI(model="gpt-4o")
    results = pd.DataFrame(columns=["Retrieved Context","Correct Documents", "Generated Answer", "Correct Answer"], index=data.index)

    # Use ThreadPoolExecutor for parallel processing
    # Gather results
    for future in tasks:
        idx, retrieved_context, generated_answer = future.result()
        results.loc[idx, "Retrieved Context"] = retrieved_context
        results.loc[idx, "Generated Answer"] = generated_answer

    results["Correct Answer"] = data.answer
    results["Correct Documents"] = data.id
    results["Golden Context"] = data.gold_inds

    results.to_csv("colpali.csv")
