import math
import pickle
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

from .constants import BM25_B, BM25_K1, stop_words_filepath
from .search_utils import (
    filter_stop_tokens,
    load_movies,
    load_stop_words,
    stem_tokens,
    tokenize_string,
)


class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Any]
    term_frequencies: dict[int, Counter[str]]
    doc_lengths: dict[int, int]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.stop_words: list[str] = load_stop_words(stop_words_filepath)


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = stem_tokens(filter_stop_tokens(tokenize_string(text), self.stop_words))
        self.doc_lengths[doc_id] = len(tokens)
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

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return float(sum(self.doc_lengths.values())) / float(len(self.doc_lengths))

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
        doc_lengths_filepath = Path(cache_dir, "doc_lengths.pkl")
        with open(index_filepath, "wb") as file:
            pickle.dump(self.index, file)
        with open(docmap_filepath, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(term_freq_filepath, "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(doc_lengths_filepath, "wb") as file:
            pickle.dump(self.doc_lengths, file)

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
        doc_lengths_filepath = Path("./cache/doc_lengths.pkl")
        if not doc_lengths_filepath.is_file():
            raise FileNotFoundError("doc_lengths file not found")
        with open(index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(term_freq_filepath, "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(doc_lengths_filepath, "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(term, 0)
        return -1

    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap)+1) / (len(self.index[term]) + 1))

    def get_bm25_idf(self, term: str) -> float:
        try:
            return math.log((len(self.docmap) - len(self.index[term]) + 0.5) / (len(self.index[term]) + 0.5) + 1)
        except KeyError:
            return 0

    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        try:
            length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        except KeyError:
            print("Invalid doc_id")
            return 0.0
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> dict[int, float]:
        query_tokens = stem_tokens(filter_stop_tokens(tokenize_string(query), self.stop_words))
        scores: dict[int, float] = {}
        for doc in self.docmap:
            total_score = 0.0
            for token in query_tokens:
                total_score += self.bm25(doc, token)
            scores[doc] = total_score
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return dict(islice(scores.items(), limit))
