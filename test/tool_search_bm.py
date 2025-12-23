from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from generated_tools import get_generated_tools


def get_search_engine(tools_desc: None | List[str] = None):
    if tools_desc is None:
        tools_desc = [t.description for t in get_generated_tools()]
    assert tools_desc is not None and len(tools_desc) > 0, "must provide tools description"
    tokenized_desc = [desc.lower().split(" ") for desc in tools_desc]

    bm25 = BM25Okapi(tokenized_desc)

    return bm25