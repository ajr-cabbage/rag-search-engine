import argparse
from lib.inverted_index import InvertedIndex
from lib.cli_commands import bm25_search_command, search_command, build_command, tf_command, idf_command,tfidf_command, bm25_idf_command, bm25_tf_command
from lib.constants import BM25_B, BM25_K1

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
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    _ = bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    _ = bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    _ = bm25search_parser.add_argument("query", type=str, help="Search query")
    _ = bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="number of results")

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
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, inv_index, k1=args.k1, b=args.b)
        case "bm25search":
            bm25_search_command(args.query, inv_index, limit=args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
