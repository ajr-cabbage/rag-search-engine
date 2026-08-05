import string
import json
import pickle
from pathlib import Path
from typing import Any
from nltk.stem import PorterStemmer

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Any]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.stop_words = load_stop_words("./data/stopwords.txt")

    def __add_document(self, doc_id, text) -> None:
        tokens = stem_tokens(filter_stop_tokens(tokenize_string(text), self.stop_words))
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
        with open(index_filepath, "wb") as file:
            pickle.dump(self.index, file)
        with open(docmap_filepath, "wb") as file:
            pickle.dump(self.docmap, file)

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

def load_stop_words(filepath: str) -> list[str]:
    with open(filepath, "r") as file:
        content = file.read()
        translate_table = str.maketrans("", "", string.punctuation)
        stop_words = content.translate(translate_table).lower().splitlines()
    return stop_words

def search_command(query: str, movies: list[Any], stop_words: list[str]) -> None:
    query_tokens = stem_tokens(filter_stop_tokens(tokenize_string(query), stop_words))
    item_number = 1
    for movie in movies:
        title_tokens = stem_tokens(filter_stop_tokens(tokenize_string(movie["title"]), stop_words))
        if has_match(query_tokens, title_tokens):
            print(f"{item_number}. {movie["title"]}")
            if item_number >= 5:
                break
            item_number += 1

def build_command(inv_index: InvertedIndex, movies_dat_filepath: str):
    inv_index.build(movies_dat_filepath)
    inv_index.save()
