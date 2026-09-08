import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify model/embedding function")
    verify_image_embedding_parser.add_argument("path", help="Path to image file")
    image_search_parser = subparsers.add_parser("image_search", help="search with image input")
    image_search_parser.add_argument("path", help="path to image file")
    image_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="max results to retrieve")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            path = args.path
            print(f"Embedding image at {path}...")
            verify_image_embedding(path)
        case "image_search":
            path = args.path
            limit = args.limit
            print(f"Searching with image at {path}...")
            image_search_command(path, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
