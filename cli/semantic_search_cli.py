import argparse

from lib.chunked_semantic_search import embed_chunks_command, search_chunked_command
from lib.semantic_search import (
    chunk_command,
    embed_query_text,
    embed_text,
    search_command,
    semantic_chunk_command,
    verify_embeddings,
    verify_model,
)


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
    chunk_parser = subparsers.add_parser("chunk", help="chunk string into n-word length segments")
    _ = chunk_parser.add_argument("text", help="the string you are chunking")
    _ = chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="words per chunk")
    _ = chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="chunk overlap parameter")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="chunk string into n-word length sentence segments")
    _ = semantic_chunk_parser.add_argument("text", help="the string you are chunking")
    _ = semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="sentences per chunk")
    _ = semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="chunk overlap parameter")
    _ = subparsers.add_parser("embed_chunks", help="embed semantic chunks")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="searched semantic chunks")
    _ = search_chunked_parser.add_argument("query", help="the query string")
    _ = search_chunked_parser.add_argument("--limit", type=int, nargs="?", default=5, help="max numer of results to return")
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
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
