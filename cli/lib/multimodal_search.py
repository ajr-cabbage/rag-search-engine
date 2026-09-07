from PIL import Image
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from typing import Any

from .search_utils import load_movies
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc["title"]}: {doc["description"]}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path: str):
        img = Image.open(img_path)
        embeddings = self.model.encode([img])
        return embeddings[0]

    def search_with_image(self, path: str, limit: int):
        img_embedding = self.embed_image(path)
        results: list[dict[str, Any]] = []
        for i, embed in enumerate(self.text_embeddings):
            cos_sim = cosine_similarity(img_embedding, embed)
            result_entry = {
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"][:100],
                "cosine_similarity": cos_sim,
            }
            results.append(result_entry)
        return sorted(results, key=lambda x: x["cosine_similarity"], reverse=True)[:limit]


def verify_image_embedding(img_path: str):
    documents = load_movies("./data/movies.json")
    mms = MultimodalSearch(documents)
    embedding = mms.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img_path: str, limit: int):
    documents = load_movies("./data/movies.json")
    mms = MultimodalSearch(documents)
    results = mms.search_with_image(img_path, limit)
    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]} (similarity: {result["cosine_similarity"]:.3f}")
        print(f"   {result["description"]}\n")
