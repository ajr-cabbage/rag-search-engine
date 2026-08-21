from ast import Invert
from collections import Counter
import math
import string
import json
import pickle
from pathlib import Path
import sys
from typing import Any
from nltk.stem import PorterStemmer

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Any]
    term_frequencies: dict[int, Counter[str]]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stop_words: list[str] = load_stop_words("./data/stopwords.txt")

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
            #if doc_id in self.term_frequencies:
            #    self.term_frequencies[doc_id].update([token])
            #else:
            #    self.term_frequencies[doc_id] = Counter()
            #    self.term_frequencies[doc_id].update([token])

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

def load_stop_words(filepath: str) -> list[str]:
    with open(filepath, "r") as file:
        content = file.read()
        translate_table = str.maketrans("", "", string.punctuation)
        stop_words = content.translate(translate_table).lower().splitlines()
    return stop_words

stop_tokens = load_stop_words("./data/stopwords.txt")

def tokenize_term(term: str) -> str:
    term_token = stem_tokens(filter_stop_tokens(tokenize_string(term), stop_tokens))
    if len(term_token) != 1:
        raise ValueError("term length is not 1")
    return term_token[0]


def tokenize_string(input: str) -> list[str]:
    if not input:
        return []
    translate_table = str.maketrans("", "", string.punctuation)
    tokens = input.translate(translate_table).lower().split()
    return [token for token in tokens if token]

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = list(map(stemmer.stem, tokens))
    return stemmed_tokens

def filter_stop_tokens(input_tokens: list[str], stop_tokens: list[str]) -> list[str]:
    if not input_tokens:
        return []
    if not stop_tokens:
        return input_tokens
    filtered_tokens = []
    for input_token in input_tokens:
        if input_token not in stop_tokens:
            filtered_tokens.append(input_token)
    return filtered_tokens

def has_match(query_tokens: list[str], ref_tokens: list[str]) -> bool:
    if not (query_tokens or ref_tokens):
       return False
    for query_token in query_tokens:
        for ref_token in ref_tokens:
            if query_token in ref_token:
                return True
    return False

def load_movies(filepath: str) -> list[Any]:
    with open(filepath, "r") as file:
        content = file.read()
        movies_dat = json.loads(content)
    return movies_dat["movies"]

def search_command(query: str, inv_index: InvertedIndex) -> None:
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    query_tokens = stem_tokens(filter_stop_tokens(tokenize_string(query), inv_index.stop_words))
    matching_ids: list[int] = []
    for query_token in query_tokens:
        if query_token in inv_index.index:
            for id in inv_index.index[query_token]:
                matching_ids.append(id)
    matching_ids.sort()
    if len(matching_ids) == 0:
        print("No Matches!")
        sys.exit(0)
    else:
        for id in matching_ids[:5]:
            print(inv_index.docmap[id]["title"])

def build_command(inv_index: InvertedIndex, movies_dat_filepath: str):
    inv_index.build(movies_dat_filepath)
    inv_index.save()

def tf_command(doc_id: int, term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(inv_index.get_tf(doc_id, tok))

def idf_command(term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(f"Inverse document frequency of '{term}': {inv_index.get_idf(tok):.2f}")

def tfidf_command(doc_id: int, term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    tfidf_score: float = inv_index.get_tf(doc_id, tok) * inv_index.get_idf(tok)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf_score:.2f}")

def bm25_idf_command(term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(f"BM25 IDF score of '{term}': {inv_index.get_bm25_idf(tok):.2f}")
