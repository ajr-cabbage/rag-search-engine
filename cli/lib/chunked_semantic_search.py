import json
from typing import Any
from pathlib import Path

from .semantic_search import SemanticSearch, semantic_chunk_text
from .search_utils import load_movies
import numpy as np
import os

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        doc_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        chunks: list[str] = []
        chunk_meta: list[dict[str, int]] = []
        for i in range(len(self.documents)):
            if not self.documents[i]["description"]:
                continue
            doc_chunks = semantic_chunk_text(self.documents[i]["description"], 4, 1)
            for n in range(len(doc_chunks)):
                chunks.append(doc_chunks[n])
                chunk_meta_entry = {
                    "movie_idx": i,
                    "chunk_idx": n,
                    "total_chunks": len(doc_chunks)
                }
                chunk_meta.append(chunk_meta_entry)
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta
        chunk_embeddings_filepath = Path("./cache/chunk_embeddings.npy")
        with open(chunk_embeddings_filepath, "wb") as file:
            np.save(file, self.chunk_embeddings)
        chunk_metadata_filepath = Path("./cache/chunk_metadata.json")
        with open(chunk_metadata_filepath, "w") as file:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        doc_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        chunk_embeddings_filepath = Path("./cache/chunk_embeddings.npy")
        chunk_metadata_filepath = Path("./cache/chunk_metadata.json")
        if os.path.exists(chunk_embeddings_filepath) and os.path.exists(chunk_metadata_filepath):
            with open(chunk_embeddings_filepath, "rb") as file:
                self.chunk_embeddings = np.load(file)
            with open(chunk_metadata_filepath, "r") as file:
                self.chunk_metadata = json.load(file)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

def embed_chunks_command():
    chunk_sem_search = ChunkedSemanticSearch()
    documents = load_movies("./data/movies.json")
    embeddings = chunk_sem_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
