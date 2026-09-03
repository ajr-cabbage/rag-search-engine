import argparse

from lib.hybrid_search import enhance_query_expand, enhance_query_rewrite, enhance_query_spell, normalize_command, rrf_search_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize", help="normalize a list of scores")
    _ = normalize_parser.add_argument("scores", type=float, nargs="*", help="list of scores to normalize")
    weighted_search_parser = subparser.add_parser("weighted-search", help="hybrid keyword/semantic search")
    _ = weighted_search_parser.add_argument("query", help="query string")
    _ = weighted_search_parser.add_argument("--alpha", nargs="?", type=float, default=0.5, help="0.0 = semantic; 1.0 = keyword")
    _ = weighted_search_parser.add_argument("--limit", nargs="?", type=int, default=5, help="max results")
    rrf_search_parser = subparser.add_parser("rrf-search", help="Reciprical Rank Fusion search")
    _ = rrf_search_parser.add_argument("query", help="search query")
    _ = rrf_search_parser.add_argument("-k", nargs="?", type=int, default=60, help="rrf weighting constant")
    _ = rrf_search_parser.add_argument("--limit", nargs="?", type=int, default=5, help="max results")
    _ = rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    _ = rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="results rerank type")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            if args.enhance == "spell":
                enhanced_query = enhance_query_spell(args.query)
                if enhanced_query != args.query:
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                    rrf_search_command(enhanced_query, args.k, args.limit)
                else:
                    rrf_search_command(args.query, args.k, args.limit)
            elif args.enhance == "rewrite":
                enhanced_query = enhance_query_rewrite(args.query)
                if enhanced_query != args.query:
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                    rrf_search_command(enhanced_query, args.k, args.limit)
                else:
                    rrf_search_command(args.query, args.k, args.limit)
            elif args.enhance == "expand":
                enhanced_query = enhance_query_expand(args.query)
                if enhanced_query != args.query:
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                    rrf_search_command(enhanced_query, args.k, args.limit)
                else:
                    rrf_search_command(args.query, args.k, args.limit)
            else:
                rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
