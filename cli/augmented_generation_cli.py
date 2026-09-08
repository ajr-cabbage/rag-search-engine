import argparse

from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summarize_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="search + summarize")
    summarize_parser.add_argument("query", type=str, help="search with query summary")
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of results to return")
    citations_parser = subparsers.add_parser("citations", help="search + summarize")
    citations_parser.add_argument("query", type=str, help="search with cited summary")
    citations_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of results to return")
    question_parser = subparsers.add_parser("question", help="search + summarize")
    question_parser.add_argument("query", type=str, help="search with answered question")
    question_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            limit = args.limit
            summarize_command(query, limit)
        case "citations":
            query = args.query
            limit = args.limit
            citations_command(query, limit)
        case "question":
            query = args.query
            limit = args.limit
            question_command(query, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
