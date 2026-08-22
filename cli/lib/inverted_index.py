import math
from collections import Counter
from pathlib import Path
import pickle
from typing import Any
from .search_utils import load_stop_words, stem_tokens, filter_stop_tokens, tokenize_string, load_movies
from .constants import BM25_K1, stop_words_filepath

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Any]
    term_frequencies: dict[int, Counter[str]]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stop_words: list[str] = load_stop_words(stop_words_filepath)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = stem_tokens(filter_stop_tokens(tokenize_string(text), self.stop_words))
        if doc_id in self.term_frequencies:
            self.term_frequencies[doc_id].update(tokens)
        else:
            self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id].update(tokens)
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set()
                self.index[token].add(doc_id)

    def get_documents(self, term) -> list[int]:
        return sorted(self.index[term])

    def build(self, filepath: str) -> None:
        movies = load_movies(filepath)
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")

    def save(self) -> None:
        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_filepath = Path(cache_dir, "index.pkl")
        docmap_filepath = Path(cache_dir, "docmap.pkl")
        term_freq_filepath = Path(cache_dir, "term_frequencies.pkl")
        with open(index_filepath, "wb") as file:
            pickle.dump(self.index, file)
        with open(docmap_filepath, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(term_freq_filepath, "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self) -> None:
        index_path = Path("./cache/index.pkl")
        if not index_path.is_file():
            raise FileNotFoundError("index file not found")
        docmap_path = Path("./cache/docmap.pkl")
        if not docmap_path.is_file():
            raise FileNotFoundError("docmap file not found")
        term_freq_filepath = Path("./cache/term_frequencies.pkl")
        if not term_freq_filepath.is_file():
            raise FileNotFoundError("term_frequencies file not found")
        with open(index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(term_freq_filepath, "rb") as file:
            self.term_frequencies = pickle.load(file)

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(term, 0)
        return -1

    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap)+1) / (len(self.index[term]) + 1))

    def get_bm25_idf(self, term: str) -> float:
        return math.log((len(self.docmap) - len(self.index[term]) + 0.5) / (len(self.index[term]) + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)
