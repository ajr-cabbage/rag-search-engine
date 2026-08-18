import argparse
from typing import Any

from nltk.sem.evaluate import sys
from lib.search_utils import build_command, search_command, load_movies, load_stop_words, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    movies_dat_filepath = "./data/movies.json"
    stop_words_filepath = "./data/stopwords.txt"
    movies_list = load_movies(movies_dat_filepath)
    stop_words = load_stop_words(stop_words_filepath)

    inv_index = InvertedIndex()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            search_command(args.query, inv_index)
        case "build":
            print(f"Building index from {movies_dat_filepath} ...")
            build_command(inv_index, movies_dat_filepath)
            # s = list(inv_index.index["merida"])
            # print(f"First document for token 'merida' = {s[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
