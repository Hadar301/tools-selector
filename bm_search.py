from time import time
from typing import List

import bm25s
import numpy as np
import Stemmer
from rank_bm25 import BM25Okapi

from prototype.tools import get_tools_description, get_tools_name

_stemmer = Stemmer.Stemmer("english")


def get_search_engine(tools_desc: None | List[str] = None):
    if tools_desc is None:
        tools_desc = get_tools_description()
    assert tools_desc is not None and len(tools_desc) > 0, "must provide tools description"
    tokenized_desc = [desc.lower().split(" ") for desc in tools_desc]

    bm25 = BM25Okapi(tokenized_desc)

    return bm25


def get_search_engine2(tools_desc: None | List[str] = None):
    if tools_desc is None:
        tools_desc = get_tools_description()
    assert tools_desc is not None and len(tools_desc) > 0, "must provide tools description"
    description_tokens = bm25s.tokenize(tools_desc, stopwords="en", stemmer=_stemmer)
    retriever = bm25s.BM25()
    retriever.index(description_tokens)

    return retriever


if __name__ == "__main__":
    tools_name = get_tools_name()
    search_engine = get_search_engine()
    search_engine2 = get_search_engine2()
    query = "what is the weather in London? multiply 2 by 4"

    print("=" * 10)
    print("rank_bm25")
    start_time = time()
    tokenized_query = query.split(" ")
    print(tokenized_query)

    scores = search_engine.get_scores(tokenized_query)
    scores = np.where(scores < 0.1, 0, scores)
    print(scores)

    tools = [t for i, t in enumerate(tools_name) if scores[i] > 0.5]
    print(tools)
    print(f"took total of: {time() - start_time:.4f} seconds")
    print("=" * 10, "\n")

    print("=" * 10)
    print("bm25s")
    start_time = time()
    query_tokens = bm25s.tokenize(query, stemmer=_stemmer)

    results, scores = search_engine2.retrieve(query_tokens, k=len(get_tools_description()))
    scores = scores[0]
    scores = np.where(scores < 0.1, 0, scores)
    print(scores)

    tools = [t for i, t in enumerate(tools_name) if scores[i] > 0.5]
    print(tools)
    print(f"took total of: {time() - start_time:.4f} seconds")
    print("=" * 10)
