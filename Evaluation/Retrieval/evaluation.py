import pandas as pd
from datasets import load_dataset
import json 
import pickle 
import os
from statistics import mean
import pytrec_eval

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

def get_expected_qrels(task):

    if task == "FinanceBench":
        data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
        data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
        qrels = {str(idx): {f"{row.doc_name}.pdf_page_{row.page_num}": 1} for idx, row in data.iterrows()}
        return qrels
    
    elif task == "FinQA":
        dataset = load_dataset("ibm/finqa", trust_remote_code=True)
        data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
        data.reset_index(drop=True, inplace=True)
        data = data[["id", "question", "answer", "gold_inds"]]
        data["Company"] = [row[0] for row in data.id.str.split("/")]
        data["Year"] = [row[1] for row in data.id.str.split("/")]
        data.id = data.id.map(lambda x: x.split("-")[0])
        qrels = {str(idx) : {str(row.id) : 1} for idx, row in data.iterrows()}
        return qrels

    elif task == "Table_VQA":
        def process_qa_id(qa_id):
            splitted = qa_id.split(".")[0]
            return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

        data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt"]]
        data.qa_id = data.qa_id.apply(process_qa_id)
        data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
        data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
        data = data.rename(columns={"qa_id": "id"})
        qrels = {str(idx) : {str(row.id) : 1} for idx, row in data.iterrows()}
        return qrels

def evaluate_pipeline(task, directory, expected_qrels):
    """Evaluate a single pipeline and return metrics."""
    ndcg_scores = {}
    mrr_scores = {}

    directory = os.path.join(parent_dir, f".results/{task}/retrieval/{directory}")

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Skipping.")
        return ndcg_scores, mrr_scores

    for qrel_file in os.listdir(directory):
        with open(os.path.join(directory, qrel_file), "r") as f:
            qrels = json.load(f)

        qrels = {k: {doc_id: score for doc_id, score in v.items()} for k, v in qrels.items() if k in expected_qrels.keys()}

        truncated_qrels = {}
        for query_id, ranking in qrels.items():
            sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
            truncated_qrels[query_id] = {doc_id: score for doc_id, score in sorted_ranking[:5]}

        evaluator = pytrec_eval.RelevanceEvaluator(expected_qrels, {'ndcg', 'recip_rank'})
        evaluation_results = evaluator.evaluate(truncated_qrels)

        total_ndcg = sum(metrics['ndcg'] for metrics in evaluation_results.values())
        total_mrr = sum(metrics['recip_rank'] for metrics in evaluation_results.values())

        average_ndcg = total_ndcg / len(evaluation_results)
        average_mrr = total_mrr / len(evaluation_results)

        ndcg_scores[qrel_file] = average_ndcg
        mrr_scores[qrel_file] = average_mrr

    return ndcg_scores, mrr_scores

def main(task):
    expected_qrels = get_expected_qrels(task=task)
    pipeline_directories = ["text", "colpali", "hybrid", "voyage"]
    
    all_results = []

    for directory in pipeline_directories:
        print(f"Evaluating pipeline: {directory}")
        ndcg, mrr = evaluate_pipeline(task, directory, expected_qrels)

        for qrel_file, score in ndcg.items():
            all_results.append({
                "Pipeline": directory,
                "File": qrel_file,
                "nDCG@5": score,
                "MRR@5": mrr.get(qrel_file, 0)
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(current_dir, f"{task}_retrieval_results.csv"), index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    tasks = ["FinQA", "Table_VQA", "FinanceBench"]
    for task in tasks:
        main(task)
