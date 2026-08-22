import sys
from .inverted_index import InvertedIndex
from .search_utils import stem_tokens, filter_stop_tokens, tokenize_string, tokenize_term
from .constants import BM25_K1

def search_command(query: str, inv_index: InvertedIndex) -> None:
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    query_tokens = stem_tokens(filter_stop_tokens(tokenize_string(query), inv_index.stop_words))
    matching_ids: list[int] = []
    for query_token in query_tokens:
        if query_token in inv_index.index:
            for id in inv_index.index[query_token]:
                matching_ids.append(id)
    matching_ids.sort()
    if len(matching_ids) == 0:
        print("No Matches!")
        sys.exit(0)
    else:
        for id in matching_ids[:5]:
            print(inv_index.docmap[id]["title"])

def build_command(inv_index: InvertedIndex, movies_dat_filepath: str):
    inv_index.build(movies_dat_filepath)
    inv_index.save()

def tf_command(doc_id: int, term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(inv_index.get_tf(doc_id, tok))

def idf_command(term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(f"Inverse document frequency of '{term}': {inv_index.get_idf(tok):.2f}")

def tfidf_command(doc_id: int, term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    tfidf_score: float = inv_index.get_tf(doc_id, tok) * inv_index.get_idf(tok)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf_score:.2f}")

def bm25_idf_command(term: str, inv_index: InvertedIndex):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(f"BM25 IDF score of '{term}': {inv_index.get_bm25_idf(tok):.2f}")

def bm25_tf_command(doc_id: int, term: str, inv_index: InvertedIndex, k1=BM25_K1):
    try:
        inv_index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        tok = tokenize_term(term)
    except ValueError as e:
        print(e)
        return
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {inv_index.get_bm25_tf(doc_id, tok):.2f}")
