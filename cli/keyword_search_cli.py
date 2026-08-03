import argparse
import json
import string
from nltk.stem import PorterStemmer

def tokenize_string(input: str) -> list[str]:
    if not input:
        return []
    translate_table = str.maketrans("", "", string.punctuation)
    tokens = input.translate(translate_table).lower().split()
    return [token for token in tokens if token]

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
        case _:
            parser.print_help()

    movies_dat = {}
    with open("./data/movies.json", "r") as file:
        content = file.read()
        movies_dat = json.loads(content)
    with open("./data/stopwords.txt", "r") as file:
        content = file.read()
        stop_words = tokenize_string(content)

    movies_list = movies_dat["movies"]
    stemmer = PorterStemmer()
    query_tokens: list[str] = list(map(stemmer.stem, filter_stop_tokens(tokenize_string(args.query), stop_words)))
    item_number = 1
    for movie in movies_list:
        title_tokens: list[str] = list(map(stemmer.stem, filter_stop_tokens(tokenize_string(movie["title"]), stop_words)))
        if has_match(query_tokens, title_tokens):
            print(f"{item_number}. {movie["title"]}")
            if item_number >= 5:
                break
            item_number += 1

if __name__ == "__main__":
    main()
