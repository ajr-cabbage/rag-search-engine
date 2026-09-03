import os
from typing import Any

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        _ = self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(Path("./cache/index.pkl")):
            self.idx.build("./data/movies.json")
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query: str, limit: int) -> dict[int, float]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict[str, Any]]:
        bm25_results = self._bm25_search(query, limit*500)
        bm25_keys = list(bm25_results)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        bm25_norm = normalize_scores(list(bm25_results.values()))
        for i in range(len(bm25_results)):
            bm25_results[bm25_keys[i]] = bm25_norm[i]
        sem_norm = normalize_scores([sem_result["score"] for sem_result in sem_results])
        combined_results: list[dict[str, Any]] = []
        for i in range(len(sem_results)):
            result_entry = {
                "doc_id": sem_results[i]["id"],
                "bm25": bm25_results[sem_results[i]["id"]],
                "semantic": sem_norm[i],
                "hybrid": hybrid_score(bm25_results[sem_results[i]["id"]], sem_norm[i], alpha),
                "document": self.idx.docmap[sem_results[i]["id"]]
            }
            combined_results.append(result_entry)
        return sorted(combined_results, key=lambda x: x["hybrid"], reverse=True)

    def rrf_search(self, query: str, k: int, limit: int) -> list[dict[str, Any]]:
        bm25_results = self._bm25_search(query, limit*500)
        bm25_keys = list(bm25_results)
        for i in range(len(bm25_results)):
            bm25_results[bm25_keys[i]] = i + 1
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        combined_results: list[dict[str,Any]] = []
        for i in range(len(sem_results)):
            bm25_rank = len(bm25_results)
            if sem_results[i]["id"] in bm25_results:
                bm25_rank = bm25_results[sem_results[i]["id"]]
            result_entry = {
                "doc_id": sem_results[i]["id"],
                "bm25_rank": bm25_rank,
                "semantic_rank": i + 1,
                "rrf_score": rrf_score(bm25_rank, k) + rrf_score(i + 1, k),
                "document": self.idx.docmap[sem_results[i]["id"]]
            }
            combined_results.append(result_entry)
        return sorted(combined_results, key=lambda x: x["rrf_score"], reverse=True)

def normalize_scores(scores:list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    if min(scores) == max(scores):
        return [1.0 for _ in range(len(scores))]
    results: list[float] = []
    for score in scores:
        results.append((score - min(scores))/(max(scores) - min(scores)))
    return results

def normalize_command(scores:list[float]):
    results = normalize_scores(scores)
    for result in results:
        print(f"* {result:.4f}")

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.weighted_search(query, alpha, limit)
    for i in range(limit):
        print(f"{i+1}. {results[i]["document"]["title"]}")
        print(f"  Hybrid Score: {results[i]["hybrid"]:.4f}")
        print(f"  BM25: {results[i]["bm25"]:.4f}, Semantic: {results[i]["semantic"]:.4f}")
        print(f"  {results[i]["document"]["description"][:100]}...")

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def rrf_search_command(query: str, k: int, limit: int):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, k, limit)
    for i in range(limit):
        print(f"{i+1}. {results[i]["document"]["title"]}")
        print(f"  RRF Score: {results[i]["rrf_score"]:.4f}")
        print(f"  BM25: {results[i]["bm25_rank"]}, Semantic: {results[i]["semantic_rank"]}")
        print(f"  {results[i]["document"]["description"][:100]}...")

def enhance_query_spell(query: str) -> str:
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    message_content = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="openrouter/free")
    return str(response.choices[0].message.content)

def enhance_query_rewrite(query: str) -> str:
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    message_content = f"""
    Rewrite the user-provided movie search query below to be more specific and searchable.
    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic
    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.
    User query: "{query}"
    """
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="openrouter/free")
    return str(response.choices[0].message.content)

def enhance_query_expand(query: str) -> str:
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    message_content = f"""
    Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="openrouter/free")
    return str(response.choices[0].message.content)
