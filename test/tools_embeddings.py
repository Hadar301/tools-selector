import torch
from typing import Dict
import sys
from generated_tools import get_generated_tools

sys.path.append('../../tools-selector')
from embeddings import embed_text

def get_tools_embeddings() -> Dict[str, torch.Tensor]:
    tools = get_generated_tools()
    tools_embed = {}
    for t in tools:
        tools_embed[t.name] = embed_text(t.description)

    return tools_embed

if __name__=="__main__":
    tools_embed = get_tools_embeddings()
    print(f"there are {len(tools_embed)} embeddings for the tools")