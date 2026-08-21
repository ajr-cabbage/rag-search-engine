import argparse
from typing import Any

from nltk.sem.evaluate import sys
from lib.search_utils import bm25_idf_command, build_command, idf_command, search_command, tf_command, InvertedIndex, tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    _ = search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Print term frequency in a doc for a given term")
    _ = tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = tf_parser.add_argument("term", type=str, help="freq match term")
    idf_parser = subparsers.add_parser("idf", help="Print inter document frequency")
    _ = idf_parser.add_argument("term", type=str, help="idf match term")
    tfidf_parser = subparsers.add_parser("tfidf", help="Print TF-IDF value")
    _ = tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = tfidf_parser.add_argument("term", type=str, help="tf-idf match term")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    _ = bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    movies_dat_filepath = "./data/movies.json"

    inv_index = InvertedIndex()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            search_command(args.query, inv_index)
        case "build":
            print(f"Building index from {movies_dat_filepath} ...")
            build_command(inv_index, movies_dat_filepath)
        case "tf":
            print(f"Returning frequency for {args.term} in {args.doc_id}")
            tf_command(args.doc_id, args.term, inv_index)
        case "idf":
            idf_command(args.term, inv_index)
        case "tfidf":
            tfidf_command(args.doc_id, args.term, inv_index)
        case "bm25idf":
            bm25_idf_command(args.term, inv_index)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
