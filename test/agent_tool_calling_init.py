import sys
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Sequence

import numpy as np
import requests
import torch
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware, wrap_model_call
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI
from tools_embeddings import get_generated_tools, get_tools_embeddings

sys.path.append("../../tools-selector")
from bm_search import get_search_engine
from embeddings import cosine_similarity, embed_text

_NUM_TOOLS: int = 30

_TOOLS_EMBEDDINGS: Dict[str, torch.Tensor] = get_tools_embeddings()
_all_tools_embeds = torch.stack(list(_TOOLS_EMBEDDINGS.values())).squeeze(1)
_all_tools = list(_TOOLS_EMBEDDINGS.keys())
_all_tools_desc = [t.description for t in get_generated_tools()[:_NUM_TOOLS]]

if _NUM_TOOLS != -1:
    _TOOLS_EMBEDDINGS = dict(list(_TOOLS_EMBEDDINGS.items())[:_NUM_TOOLS])

_SEARCH_ENGINE = get_search_engine(tools_desc=_all_tools_desc)

_BASE_URL = "http://localhost:1234/v1"
_API_KEY = "lm-studio"
_LLM_MODEL = "meta-llama-3.1-8b-instruct-128k"
_SYSTEM_PROMPT = f"""
    You are a tool-using AI Agent.
    
    Before answering any user request, you must:
    1. Analyze the user query.
    2. Decide which tools (if any) are needed.
    3. Explicitly state **which tools you plan to use and why**, in a section called "Tool Plan".
    4. Only after presenting your plan, execute the tool calls in the correct order.

    Format your response exactly as follows:

    ---
    Tool Plan:
    - <ToolName1>: <Why this tool is needed>
    - <ToolName2>: <Why this tool is needed>
    (If no tools are needed: "No tools required")

    Execution:
    <tool calls in the order they should run>
    ---

    Rules:
    - Always list every tool you intend to use.
    - Never execute a tool without first mentioning it in the Tool Plan.
    - Respond with a Tool Plan even if only one tool is required.
    - If a user asks for multiple tasks, break them down and match each to the appropriate tool.
    - If the user’s request cannot be satisfied by any tool, explain why in the Tool Plan and answer without tools.

    Avialable tools: {get_generated_tools()[:_NUM_TOOLS]}
"""


### --- test connection before starting ---
response = requests.get(f"{_BASE_URL}/models", timeout=10)
assert response.status_code == 200, (
    f"{_BASE_URL} appears like it is not active, check your connection/ LLM serving"
)


# --- Initialize Local LLM (LM Studio) ---
_llm = ChatOpenAI(
    base_url=_BASE_URL,
    api_key=_API_KEY,
    model=_LLM_MODEL,
    temperature=0.42,
)

# --- MiddleWare funtionality ---
middleware = LLMToolSelectorMiddleware(model=_llm, system_prompt=_SYSTEM_PROMPT)


def get_human_content(messages) -> Sequence[str]:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content


def process_scores(scores: np.ndarray, min_keep_thres: float = 0.1) -> np.ndarray:
    scores = np.where(scores < min_keep_thres, 0, scores)  # filter scores with "low energy"

    if not np.all(scores == 0):
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = scores
    return normalized_scores


@lru_cache(maxsize=128)
def find_top_tools_embeddings(human_message: str, threshold: float = 0.25) -> List[str]:
    # all_tools_embeds = torch.stack(list(_TOOLS_EMBEDDINGS.values())).squeeze(1)
    # all_tools = list(_TOOLS_EMBEDDINGS.keys())
    human_message_embed = embed_text(human_message)
    # print(human_message_embed.shape, all_tools_embeds.shape)

    similarity_scores = cosine_similarity(human_message_embed, _all_tools_embeds)
    # print(similarity_scores)
    normalized_scores = process_scores(similarity_scores.numpy())
    # print(normalized_scores)
    indices = np.where(normalized_scores >= threshold)[0]
    return [_all_tools[i] for i in indices.tolist()]


@lru_cache(maxsize=128)
def find_top_tools_bm_search(human_message: str, threshold: float = 0.25) -> List[str]:
    human_message_tokenized = human_message.split(" ")
    scores = _SEARCH_ENGINE.get_scores(human_message_tokenized)
    # print(f"total scores: {len(scores)} max score: {np.max(scores)}")
    normalized_scores = process_scores(scores)

    indices = np.where(normalized_scores >= threshold)[0]
    # print(f"all idexes: {indices}")
    return [_all_tools[i] for i in indices.tolist()]


def filter_request_tools(
    request_tools: List[StructuredTool], relevant_tools: List[str]
) -> List[StructuredTool]:
    return [t for t in get_generated_tools()[:_NUM_TOOLS] if t.name in set(relevant_tools)]


@wrap_model_call
def filter_tools(request, handler):
    messages = getattr(request, "messages", [])
    human_message = get_human_content(messages)
    # top_tools = find_top_tools_embeddings(human_message)
    top_tools = find_top_tools_bm_search(human_message)
    req_tools = getattr(request, "tools", [])
    filtered_request_tools = filter_request_tools(req_tools, top_tools)
    setattr(request, "tools", filtered_request_tools)
    # req_tools = getattr(request, "tools", [])

    return handler(request)


@wrap_model_call  # type: ignore[misc]
async def filter_tools_async(request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
    messages = getattr(request, "messages", [])
    human_message = get_human_content(messages)
    # top_tools = find_top_tools_embeddings(human_message)
    top_tools = find_top_tools_bm_search(human_message, threshold=0.7)
    # print(len(top_tools))
    req_tools = getattr(request, "tools", [])
    filtered_request_tools = filter_request_tools(req_tools, top_tools)
    # setattr(request, "tools", filtered_request_tools)
    request.override(tools=filtered_request_tools)

    return await handler(request)


def initialize_filtered_agent(max_tools: int) -> AgentState:
    middleware_array = [filter_tools_async]  # , middleware
    return create_agent(
        _llm,
        tools=get_generated_tools()[:max_tools],
        middleware=middleware_array,
        system_prompt=_SYSTEM_PROMPT,
    )


def initialize_agent(max_tools: int) -> AgentState:
    return create_agent(
        _llm,
        tools=get_generated_tools()[:max_tools],
        # middleware=[middleware],
        system_prompt=_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    # agent_executor = initialize_agent(max_tools=_NUM_TOOLS)
    agent_executor = initialize_filtered_agent(max_tools=_NUM_TOOLS)

    query = "What are the current game giveaways on GamerPower?"

    response = agent_executor.invoke({"messages": [("user", query)]})
    # print(response["messages"][-1].content)
    for message in response["messages"]:
        if hasattr(message, "tool_calls"):
            tool_call = getattr(message, "tool_calls")
            print(tool_call)
            for tc in tool_call:
                print(tc["name"])
                # if hasattr(tc, 'name'):
                # print(tc['name'])
