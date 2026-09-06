import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .hybrid_search import HybridSearch
from .search_utils import load_movies


def rag_command(query: str):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, 60, 5)
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    results_string = json.dumps(results)
    message_content = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {results_string}

    Answer:"""
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="minimax/minimax-m3:free")
    rag_response = response.choices[0].message.content
    print("Search Results:")
    for i, doc in enumerate(results):
        print(f"- {doc["document"]["title"]}")
    print("\nRAG Response:")
    print(rag_response)

def summarize_command(query: str, limit: int):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, 60, limit)
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    results_string = json.dumps(results)
    message_content = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {results_string}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="minimax/minimax-m3:free")
    summary = response.choices[0].message.content
    print("Search Results:")
    for i, doc in enumerate(results):
        print(f"  - {doc["document"]["title"]}")
    print("\nLLM Response:")
    print(summary)

def citations_command(query: str, limit: int):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, 60, limit)
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    results_string = json.dumps(results)
    message_content = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {results_string}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="minimax/minimax-m3:free")
    summary = response.choices[0].message.content
    print("Search Results:")
    for i, doc in enumerate(results):
        print(f"  - {doc["document"]["title"]}")
    print("\nLLM Answer:")
    print(summary)

def question_command(query: str, limit: int):
    documents = load_movies("./data/movies.json")
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, 60, limit)
    _ = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    results_string = json.dumps(results)
    message_content = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {query}

    Documents:
    {results_string}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""
    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]
    response = client.chat.completions.create(messages=messages, model="minimax/minimax-m3:free")
    summary = response.choices[0].message.content
    print("Search Results:")
    for i, doc in enumerate(results):
        print(f"  - {doc["document"]["title"]}")
    print("\nAnswer:")
    print(summary)
