import os
import evaluate
import pandas as pd
from datasets import load_dataset
import json
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from collections import Counter
import re
from io import StringIO
import asyncio
import logging

# Set up current and parent directories
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))

with open(os.path.join(parent_dir, ".keys/openai_api_key.txt"), "r") as file:
    openai_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

correctness_metric = GEval(
    name="Correctness",
    criteria="Evaluate the factual accuracy of the actual output by considering both the final answer and the reasoning steps leading to it. Assess whether the intermediate steps align with the given context and contribute logically to the correct answer.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    verbose_mode=False,
)

# Set up logging
logging.basicConfig(
    filename=f"{current_dir}/generation-evaluation.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

exact_match = evaluate.load("exact_match")
f1 = evaluate.load("f1")
meteor = evaluate.load("meteor")
CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}  

def html_to_string(html_string):
    df = pd.read_html(StringIO(html_string))[0]
    df.set_index(0, inplace=True)
    df.columns = df.iloc[0]
    df = df[1:]
    df.index.name = None
    return df.to_string()

def normalize_numeric_token(token):
    token = "".join(ch for ch in token if ch not in CURRENCY_SYMBOLS)
    token = token.replace(",", "")
    try:
        float_val = float(token)
        token = str(float_val)
    except ValueError:
        pass
    
    return token

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", text)
    return tokens

def domain_normalize_tokens(tokens):
    normalized_tokens = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            t = normalize_numeric_token(t)
        normalized_tokens.append(t)
    return normalized_tokens

def compute_token_f1(prediction, reference):
    pred_text = preprocess_text(prediction)
    ref_text = preprocess_text(reference)

    pred_tokens = tokenize(pred_text)
    ref_tokens = tokenize(ref_text)

    # Domain normalization (numeric normalization, currency removal, etc.)
    pred_tokens = domain_normalize_tokens(pred_tokens)
    ref_tokens = domain_normalize_tokens(ref_tokens)

    common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common_tokens.values())

    precision = num_common / len(pred_tokens) if pred_tokens else 0.0
    recall = num_common / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def prepare_data(task):

    if task == "FinanceBench":
        data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
        data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
        data["evidence"] = data["evidence"].apply(lambda x: x[0]["evidence_text"])
        return data
    
    elif task == "FinQA":
        dataset = load_dataset("ibm/finqa", trust_remote_code=True)
        data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
        data.reset_index(drop=True, inplace=True)
        data = data[["id", "question", "answer", "gold_inds"]]
        data["Company"] = [row[0] for row in data.id.str.split("/")]
        data["Year"] = [row[1] for row in data.id.str.split("/")]
        data.id = data.id.map(lambda x: x.split("-")[0])
        return data

    elif task == "Table_VQA":
        def process_qa_id(qa_id):
            splitted = qa_id.split(".")[0]
            return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

        data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt", "text_html_table"]]
        data.qa_id = data.qa_id.apply(process_qa_id)
        data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
        data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
        data = data.rename(columns={"qa_id": "id", "gt": "answer", "text_html_table": "evidence"})
        return data

async def evaluate_query_async(query_idx, row, generations, subfolder, g_eval_scores, simple_metrics):
    reasoning = generations[str(query_idx)]["reasoning"]
    answer = generations[str(query_idx)]["answer"]
    golden_context = row["evidence"] 
    actual_output = reasoning + "\n\n" + answer
    expected_output = row["answer"]

    # G-Eval for Correctness
    def run_g_eval():
        test_case = LLMTestCase(
            input=golden_context,
            actual_output=actual_output,
            expected_output=expected_output
        )
        correctness_metric.measure(test_case)
        return {
            "G-Eval Score": correctness_metric.score,
            "G-Eval Reasoning": correctness_metric.reason,
        }

    g_eval_result = await asyncio.to_thread(run_g_eval)
    g_eval_scores.append({
        "Subfolder": subfolder,
        "Index": query_idx,
        "G-Eval Score": g_eval_result["G-Eval Score"],
        "G-Eval Reasoning": g_eval_result["G-Eval Reasoning"],
    })

    # Compute simple metrics
    exact_result = exact_match.compute(predictions=[answer], references=[expected_output])
    f1_score = compute_token_f1(answer, expected_output)
    meteor_result = meteor.compute(predictions=[preprocess_text(actual_output)], references=[preprocess_text(expected_output)])

    simple_metrics.append({
        "Subfolder": subfolder,
        "Index": query_idx,
        "Exact Match": exact_result["exact_match"],
        "F1-Score": f1_score,
        "METEOR": meteor_result["meteor"],
    })

    logging.info(f"Done with query {query_idx}")


async def evaluate_generation(task, generation_folder):
    data = prepare_data(task)

    for subfolder in ["text", "image"]:
        subfolder_path = os.path.join(generation_folder, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping.")
            continue

        answers_file = os.path.join(subfolder_path, "answers.json")
        if not os.path.exists(answers_file):
            print(f"Answers file {answers_file} does not exist. Skipping.")
            continue

        with open(answers_file, "r") as f:
            generations = json.load(f)

        g_eval_scores = []
        simple_metrics = []

        tasks = [
            evaluate_query_async(idx, row, generations, subfolder, g_eval_scores, simple_metrics)
            for idx, row in data.iterrows()
        ]

        await asyncio.gather(*tasks)

        # Save G-Eval Results
        g_eval_df = pd.DataFrame(g_eval_scores)
        g_eval_df.to_csv(os.path.join(subfolder_path, f"{current_dir}/{task}_{subfolder}_g_eval.csv"), index=False)

        # Save Simple Metrics
        simple_df = pd.DataFrame(simple_metrics)
        simple_df.to_csv(os.path.join(subfolder_path, f"{current_dir}/{task}_{subfolder}_simple_metrics.csv"), index=False)

        print(f"Results saved for {task} in {subfolder_path}")

if __name__ == "__main__":
    tasks = ["FinanceBench"]

    async def main():
        for task in tasks:
            generation_folder = os.path.join(parent_dir, f".results/{task}/generation")
            await evaluate_generation(task, generation_folder)

    asyncio.run(main())
