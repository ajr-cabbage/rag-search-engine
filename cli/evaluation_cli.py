import argparse
import json

from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("./data/golden_dataset.json", "r") as file:
        golden_dataset = json.load(file)
    test_cases = golden_dataset["test_cases"]
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)

    print("k=", limit)

    for test in test_cases:
        results = hs.rrf_search(test["query"], 60, limit)
        result_titles = [result["document"]["title"] for result in results]
        intersection = list(set(result_titles) & set(test["relevant_docs"]))
        precision = len(intersection) / limit
        recall = len(intersection) / len(test["relevant_docs"])
        f1 = 0
        if precision + recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {test["query"]}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {result_titles}")
        print(f"  - Relevant: {test["relevant_docs"]}\n")

if __name__ == "__main__":
     main()
