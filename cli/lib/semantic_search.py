from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import os
from torch.nn.functional import embedding

from .search_utils import load_movies

class SemanticSearch:

    model: SentenceTransformer
    embeddings: Any
    documents: list[Any]
    document_map: dict[int, Any]

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = []
        self.document_map = {}

    def generate_embedding(self, text: str) -> Any:
        if not text or text.isspace():
            raise ValueError("Empty/whitepace string")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embedddings(self, documents: list[Any]):
        self.documents = documents
        doc_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        embeddings_filepath = Path("./cache/movie_embeddings.npy")
        with open(embeddings_filepath, "wb") as file:
            np.save(file, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Any]):
        self.documents = documents
        doc_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        embeddings_filepath = Path("./cache/movie_embeddings.npy")
        if os.path.exists(embeddings_filepath):
            with open(embeddings_filepath, "rb") as file:
                self.embeddings = np.load(file)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embedddings(documents)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores: list[tuple[float, Any]] = []
        for i in range(len(self.embeddings)):
            cos_sim: float = cosine_similarity(query_embedding, self.embeddings[i])
            score_entry = (cos_sim, self.documents[i])
            scores.append(score_entry)
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score in sorted_scores:
            results_entry = {
                "score": score[0],
                "title": score[1]["title"],
                "description": score[1]["description"]
            }
            results.append(results_entry)
        return results[:limit]



def verify_model():
    test_model = SemanticSearch()
    print(f"Model loaded: {test_model.model}")
    print(f"Max sequence length: {test_model.model.max_seq_length}")

def embed_text(text: str):
    sem_search = SemanticSearch()
    try:
        embedding = sem_search.generate_embedding(text)
    except ValueError as e:
        print(e)
        return
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_search = SemanticSearch()
    documents: list[Any] = load_movies("./data/movies.json")
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int=5):
    sem_search = SemanticSearch()
    documents = load_movies("./data/movies.json")
    sem_search.load_or_create_embeddings(documents)
    results = sem_search.search(query, limit)
    for i in range(len(results)):
        print(f"{i+1}. {results[i]["title"]} (score: {results[i]["score"]:.4f})\n  {results[i]["description"]}\n")
