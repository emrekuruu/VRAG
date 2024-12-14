import json
import os
import numpy as np

def load_qrels(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_qrels(qrels, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(qrels, f, indent=4)

def normalize_scores(scores, type_ = "min-max"):
    if type_ == "z-score":
        if not scores:
            return {}
        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values) or 1 
        return {doc_id: (score - mean) / std for doc_id, score in scores.items()}
    else:
        min_score = min(scores.values())
        max_score = max(scores.values())
        range_score = max_score - min_score or 1  # Avoid division by zero
        return {doc_id: (score - min_score) / range_score for doc_id, score in scores.items()}

def combine_qrels(vanilla_qrels, colpali_qrels, top_k=5, alpha=0.6, beta=0.4):
    combined_qrels = {}

    for query_id in set(vanilla_qrels.keys()).union(colpali_qrels.keys()):
        vanilla_results = vanilla_qrels.get(query_id, {})
        colpali_results = colpali_qrels.get(query_id, {})

        # Normalize scores for both pipelines
        vanilla_scores = normalize_scores(vanilla_results)
        colpali_scores = normalize_scores(colpali_results)

        # Combine scores with weights
        combined_scores = {}
        for doc_id in set(vanilla_scores.keys()).union(colpali_scores.keys()):
            vanilla_score = vanilla_scores.get(doc_id, 0)
            colpali_score = colpali_scores.get(doc_id, 0)
            combined_scores[doc_id] = alpha * vanilla_score + beta * colpali_score

        # Sort combined scores and keep only the top_k results
        top_combined_scores = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )
        combined_qrels[query_id] = top_combined_scores

    return combined_qrels

def main(vanilla_file, colpali_file, output_file):
    vanilla_qrels = load_qrels(vanilla_file)
    colpali_qrels = load_qrels(colpali_file)

    alpha = 0.4
    beta = 1 - alpha  

    hybrid_qrels = combine_qrels(vanilla_qrels, colpali_qrels, alpha=alpha, beta=beta)

    save_qrels(hybrid_qrels, output_file)
    print(f"Hybrid Qrels saved to {output_file}")

if __name__ == "__main__":
    vanilla_file = "results/colpali_qrels.json"
    colpali_file = "results/voyage_qrels.json"
    output_file = "results/hybrid_qrels.json"
    
    main(vanilla_file, colpali_file, output_file)
