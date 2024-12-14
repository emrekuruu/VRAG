import json
import os
import numpy as np

# Utility functions
def load_qrels(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_qrels(qrels, file_path):
    directory = os.path.dirname(file_path)
    if directory: 
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(qrels, f, indent=4)

def normalize_scores(scores, type_="min-max"):
    if not scores:
        return {}
    values = list(scores.values())

    if type_ == "z-score":
        mean = np.mean(values)
        std = np.std(values) or 1  # Avoid division by zero
        return {doc_id: (score - mean) / std for doc_id, score in scores.items()}
    elif type_ == "min-max":
        min_score = min(values)
        max_score = max(values)
        range_score = max_score - min_score or 1  # Avoid division by zero
        return {doc_id: (score - min_score) / range_score for doc_id, score in scores.items()}

# Pipeline 1: Z-Score Normalization Hybridization
def pipeline_zscore(first_qrels, second_qrels, alpha, beta, top_k=5):
    combined_qrels = {}

    for query_id in set(first_qrels.keys()).union(second_qrels.keys()):
        first_results = first_qrels.get(query_id, {})
        second_results = second_qrels.get(query_id, {})

        # Normalize scores using z-score
        first_scores = normalize_scores(first_results, type_="z-score")
        second_scores = normalize_scores(second_results, type_="z-score")

        # Combine scores with weights
        combined_scores = {}
        for doc_id in set(first_scores.keys()).union(second_scores.keys()):
            first_score = first_scores.get(doc_id, 0)
            second_score = second_scores.get(doc_id, 0)
            combined_scores[doc_id] = alpha * first_score + beta * second_score

        # Sort combined scores and keep only the top_k results
        combined_qrels[query_id] = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )

    return combined_qrels

# Pipeline 2: Min-Max Normalization Hybridization
def pipeline_minmax(first_qrels, second_qrels, alpha, beta, top_k=5):
    combined_qrels = {}

    for query_id in set(first_qrels.keys()).union(second_qrels.keys()):
        first_results = first_qrels.get(query_id, {})
        second_results = second_qrels.get(query_id, {})

        # Normalize scores using min-max
        first_scores = normalize_scores(first_results, type_="min-max")
        second_scores = normalize_scores(second_results, type_="min-max")

        # Combine scores with weights
        combined_scores = {}
        for doc_id in set(first_scores.keys()).union(second_scores.keys()):
            first_score = first_scores.get(doc_id, 0)
            second_score = second_scores.get(doc_id, 0)
            combined_scores[doc_id] = alpha * first_score + beta * second_score

        # Sort combined scores and keep only the top_k results
        combined_qrels[query_id] = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )

    return combined_qrels

# Pipeline 3: Rank-Based Weighted Hybridization
def pipeline_rankbased(first_qrels, second_qrels, alpha, beta, top_k=5):
    combined_qrels = {}

    for query_id in set(first_qrels.keys()).union(second_qrels.keys()):
        first_results = first_qrels.get(query_id, {})
        second_results = second_qrels.get(query_id, {})

        # Sort Qrels values by relevance
        first_sorted = dict(sorted(first_results.items(), key=lambda item: item[1], reverse=True))
        second_sorted = dict(sorted(second_results.items(), key=lambda item: item[1], reverse=True))

        # Create rank mappings for both pipelines
        first_rankings = {doc_id: rank + 1 for rank, doc_id in enumerate(first_sorted)}
        second_rankings = {doc_id: rank + 1 for rank, doc_id in enumerate(second_sorted)}

        # Max rank normalization (lower rank = higher importance)
        max_first_rank = max(first_rankings.values(), default=1)
        max_second_rank = max(second_rankings.values(), default=1)

        combined_scores = {}
        for doc_id in set(first_rankings.keys()).union(second_rankings.keys()):
            first_rank_score = 1 - (first_rankings.get(doc_id, max_first_rank) / max_first_rank)
            second_rank_score = 1 - (second_rankings.get(doc_id, max_second_rank) / max_second_rank)

            # Apply weights and combine
            combined_scores[doc_id] = alpha * first_rank_score + beta * second_rank_score

        # Sort combined scores and keep only the top_k results
        combined_qrels[query_id] = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )

    return combined_qrels

# Main function to execute all pipelines for multiple alphas
def main(first_method, second_method, first_file, second_file, output_dir):
    # Load Qrels
    first_qrels = load_qrels(first_file)
    second_qrels = load_qrels(second_file)

    # Alpha values
    alphas = np.arange(0.5, 0.85, 0.05)

    for alpha in alphas:
        beta = 1 - alpha

        # Run all pipelines
        hybrid_qrels_zscore = pipeline_zscore(first_qrels, second_qrels, alpha, beta)
        hybrid_qrels_minmax = pipeline_minmax(first_qrels, second_qrels, alpha, beta)
        hybrid_qrels_rankbased = pipeline_rankbased(first_qrels, second_qrels, alpha, beta)

        # Save results for each alpha
        save_qrels(
            hybrid_qrels_zscore,
            f"results/hybrid/{first_method}_{second_method}_zscore_qrels_{alpha:.2f}.json",
        )

        save_qrels(
            hybrid_qrels_minmax,
            f"results/hybrid/{first_method}_{second_method}_minmax_qrels_{alpha:.2f}.json",
        )
        save_qrels(
            hybrid_qrels_rankbased,
            f"results/hybrid/{first_method}_{second_method}_rank_qrels_{alpha:.2f}.json",
        )

        print(f"Saved hybrid Qrels for alpha={alpha:.2f}")

if __name__ == "__main__":

    second_method = "colpali"
    first_method = "voyage"

    second_file = "results/colpali_qrels.json"
    first_file = "results/voyage_qrels.json"

    output_dir = "result/hybrid/"

    main(first_method, second_method, first_file, second_file, output_dir)
