import sys
from functools import lru_cache
from typing import Dict, List, Sequence

import torch
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware, wrap_model_call
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI

sys.path.append("../../tools-selector")
from tools import get_tools_array, get_tools_embeddings, get_tools_name

from embeddings import cosine_similarity, embed_text

_TOOLS_EMBEDDINGS: Dict[str, torch.Tensor] = get_tools_embeddings()


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

    Avialable tools: {get_tools_array()}
"""

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


def get_ai_messages(messages):
    for m in messages:
        if isinstance(m, AIMessage):
            print(m.content)
            # return m.content


def process_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = torch.where(scores < 0.1, 0, scores)

    if not torch.all(scores == 0):
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = scores
    return normalized_scores


@lru_cache(maxsize=128)
def find_top_tools(human_message: Sequence[str], threshold: float = 0.25) -> List[str]:
    all_tools_embeds = torch.stack(list(_TOOLS_EMBEDDINGS.values())).squeeze(1)
    all_tools = get_tools_name()
    human_message_embed = embed_text(human_message)
    # print(human_message_embed.shape, all_tools_embeds.shape)

    similarity_scores = cosine_similarity(human_message_embed, all_tools_embeds)
    # print(similarity_scores)
    normalized_scores = process_scores(similarity_scores)
    # print(normalized_scores)
    indices = torch.where(normalized_scores >= threshold)[0]
    return [all_tools[i] for i in indices.tolist()]


def filter_request_tools(
    request_tools: List[StructuredTool], relevant_tools: List[str]
) -> List[StructuredTool]:
    return [t for t in request_tools if t.name in relevant_tools]


@wrap_model_call
def filter_tools(request, handler):
    # print(type(request))

    # print(request.__dict__.keys(), "\n")
    # print(request.__dict__['messages'], "\n")
    # print(request.__dict__['tools'], "\n")
    # print(request.__dict__['tool_choice'], "\n")
    # print(request)
    messages = getattr(request, "messages", [])
    # get_ai_messages(messages)
    human_message = get_human_content(messages)
    top_tools = find_top_tools(human_message)
    # print(human_message)
    # print(f"relevat tools for the task: {top_tools} \n")
    req_tools = getattr(request, "tools", [])
    # print(len(req_tools))
    filtered_request_tools = filter_request_tools(req_tools, top_tools)
    setattr(request, "tools", filtered_request_tools)
    req_tools = getattr(request, "tools", [])
    # print(len(filtered_request_tools))

    return handler(request)


def initialize_filtered_agent() -> AgentState:
    middleware_array = [filter_tools]  # , middleware
    return create_agent(
        _llm,
        tools=get_tools_array(),
        middleware=middleware_array,
        system_prompt=_SYSTEM_PROMPT,
    )


def initialize_agent() -> AgentState:
    return create_agent(
        _llm,
        tools=get_tools_array(),
        # middleware=[middleware],
        system_prompt=_SYSTEM_PROMPT,
    )
