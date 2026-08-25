import argparse
from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, search_command, verify_embeddings, verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="generate embedding for string")
    _ = embed_text_parser.add_argument("text", help="the text to embed")
    _ = subparsers.add_parser("verify_embeddings", help="verify embeddings are built correctly")
    embed_query_parser = subparsers.add_parser("embed_query", help="generate embedding for a query string")
    _ = embed_query_parser.add_argument("query", help="query string ton embed")
    search_parser = subparsers.add_parser("search", help="semantic search for query string")
    _ = search_parser.add_argument("query", help="search query")
    _ = search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
