from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from prototype.tools import get_tools_description, get_tools_name


def get_search_engine(tools_desc: None | List[str] = None):
    if tools_desc is None:
        tools_desc = get_tools_description()
    assert tools_desc is not None and len(tools_desc) > 0, "must provide tools description"
    tokenized_desc = [desc.lower().split(" ") for desc in tools_desc]

    bm25 = BM25Okapi(tokenized_desc)

    return bm25


if __name__ == "__main__":
    tools_name = get_tools_name()
    search_engine = get_search_engine()

    query = "what is the weather in London? multiply 2 by 4"
    tokenized_query = query.split(" ")
    print(tokenized_query)

    scores = search_engine.get_scores(tokenized_query)
    scores = np.where(scores < 0.1, 0, scores)
    print(scores)

    tools = [t for i, t in enumerate(tools_name) if scores[i] > 0.5]
    print(tools)
