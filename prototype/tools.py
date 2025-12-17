import sys
from typing import Dict, Sequence

import torch
from langchain_core.tools import BaseTool, tool

sys.path.append("../../tools-selector")
from embeddings import embed_text


# --- Define Tools ---
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a specific city.
    """
    print("\nGetting weather\n")
    return f"The weather in {city} is currently cloudy and 18°C."


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print("\nMultiplying\n")
    return a * b


@tool
def stock_price(stock: str) -> str:
    """
    Get the current stock price for a specific stock.
    """
    print("\nGetting Stock Price\n")
    return f"The price of stock {stock} is 456.87 USD"


@tool
def get_traffic(city: str) -> str:
    """
    Get the traffic for a specific city.
    """
    print("\nGetting weather\n")
    return f"The traffic in {city} is currently very busy."


def get_tools_array() -> Sequence[BaseTool]:
    return [get_weather, multiply, stock_price, get_traffic]


def get_tools_description() -> Sequence[str]:
    tools = get_tools_array()

    return [t.description for t in tools]


def get_tools_name() -> Sequence[str]:
    tools = get_tools_array()

    return [t.name for t in tools]


def get_tools_embeddings() -> Dict[str, torch.Tensor]:
    tools = get_tools_array()
    tools_embed = {}
    for t in tools:
        tools_embed[t.name] = embed_text(t.description)

    return tools_embed


if __name__ == "__main__":
    tools_embeddings = get_tools_embeddings()
    print(tools_embeddings.values())
