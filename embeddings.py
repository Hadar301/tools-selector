import os

import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
_MODEL = AutoModel.from_pretrained(_MODEL_NAME)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# print(help(_TOKENIZER.__call__))


def embed_text(txt: str) -> torch.Tensor:
    inputs = _TOKENIZER(txt, return_tensors="pt", truncation=True, padding=True)
    with torch.inference_mode():
        outputs = _MODEL(**inputs).last_hidden_state
    embeddings = outputs.mean(dim=1)
    return embeddings


def cosine_similarity(embed_vec1: torch.Tensor, embed_vec2: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(embed_vec1, embed_vec2)


if __name__ == "__main__":
    text1 = "hello, how are you?"
    embds1 = embed_text(text1)

    text2 = "hello, how's your day?"
    embds2 = embed_text(text2)

    similarity_score = cosine_similarity(embds1, embds2)
    print(similarity_score.shape, similarity_score)

    all_embeds = torch.stack([embds1, embds2])
    all_embeds = all_embeds.squeeze(1)
    print(all_embeds.shape)
    similarity_score = cosine_similarity(embds1, all_embeds)
    print(similarity_score.shape, similarity_score)
