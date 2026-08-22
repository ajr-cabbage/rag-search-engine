import string
import json
from pathlib import Path
import sys
from typing import Any
from nltk.stem import PorterStemmer

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
