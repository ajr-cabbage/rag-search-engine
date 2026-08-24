import argparse
from lib.semantic_search import SemanticSearch, embed_text, verify_embeddings, verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="generate embedding for string")
    _ = embed_text_parser.add_argument("text", help="the text to embed")
    _ = subparsers.add_parser("verify_embeddings", help="verify embeddings are built correctly")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
