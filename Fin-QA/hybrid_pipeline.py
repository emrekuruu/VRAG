import json
import os

def load_qrels(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_qrels(qrels, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(qrels, f, indent=4)

def combine_qrels(vanilla_qrels, colpali_qrels, top_k = 5):
    combined_qrels = {}

    # Iterate over all query IDs
    for query_id in set(vanilla_qrels.keys()).union(colpali_qrels.keys()):
        # Retrieve results from each pipeline
        vanilla_results = vanilla_qrels.get(query_id, {})
        colpali_results = colpali_qrels.get(query_id, {})

        # Normalize ranks for both pipelines
        vanilla_rankings = {doc_id: rank + 1 for rank, doc_id in enumerate(vanilla_results)}
        colpali_rankings = {doc_id: rank + 1 for rank, doc_id in enumerate(colpali_results)}

        # Max rank normalization (lower rank = higher importance)
        max_vanilla_rank = max(vanilla_rankings.values(), default=1)
        max_colpali_rank = max(colpali_rankings.values(), default=1)

        # Combine scores based on normalized rank
        combined_scores = {}
        for doc_id in set(vanilla_rankings.keys()).union(colpali_rankings.keys()):
            vanilla_score = 1 - (vanilla_rankings.get(doc_id, max_vanilla_rank) / max_vanilla_rank)
            colpali_score = 1 - (colpali_rankings.get(doc_id, max_colpali_rank) / max_colpali_rank)
            combined_scores[doc_id] = vanilla_score + colpali_score

        # Sort combined scores and keep only the top_k results
        top_combined_scores = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )
        combined_qrels[query_id] = top_combined_scores

    return combined_qrels

def main(vanilla_file, colpali_file, output_file):
    # Load Qrels
    vanilla_qrels = load_qrels(vanilla_file)
    colpali_qrels = load_qrels(colpali_file)

    # Combine Qrels
    hybrid_qrels = combine_qrels(vanilla_qrels, colpali_qrels)

    # Save the combined Qrels
    save_qrels(hybrid_qrels, output_file)
    print(f"Hybrid Qrels saved to {output_file}")


if __name__ == "__main__":
    vanilla_file = "results/vanilla_qrels.json"
    colpali_file = "results/colpali_qrels.json"
    output_file = "results/hybrid_qrels.json"
    
    # Combine Qrels
    main(vanilla_file, colpali_file, output_file)
