# Tool Selection for LLM Agents

### Reaserach Question:
This repo aims to answer the question - "Can we improve the tool selection process of an LLM-based agent using search techniques?"

This repository explores efficient tool selection for LLM-based agents, with a focus on narrowing the available tool set before an agent decides which tools to call.
The project evaluates embedding-based retrieval and LLM-based verification strategies for selecting the correct tools from a tool catalog, emphasizing multi-tool requests and agent control-flow correctness.
The scope is intentionally limited to tool reasoning and selection, not tool execution.

## Motivation

As LLM agents are integrated with an increasing number of tools, exposing the full tool catalog to the model becomes:

* Inefficient (larger prompts, higher inference cost)
* Error-prone (irrelevant tool calls and hallucinations)
* Difficult to control (weak separation of responsibilities)

This project investigates whether pre-selecting a small, relevant subset of tools—based on tool descriptions and user intent—can:

* Improve agent reliability
* Reduce unnecessary tool consideration
* Preserve correctness for both single-tool and multi-tool requests

## Key Ideas
* Embedding-based tool retrieval with high recall
* Middleware-level enforcement of tool visibility in LangChain agents
* Evaluation without executing real APIs

## Dataset
This project uses xLAM-function-calling-60k, a synthetic open-source dataset for evaluating function calling and tool usage.

### Why xLAM?
* Tools are abstract and cost-free
* Tool definitions include names, descriptions, and schemas
* Ground-truth tool calls are provided
* Supports both single-tool and multi-tool prompts
* The dataset enables deterministic and reproducible evaluation without requiring any API credentials.

## Metrics
Among the important datapoints that were collected are:
1. Accuracy - cacluated by the overlap of the expected tools to the actually tools used devided by the the maximum of them.
``` python3
overlap = len(set(expected_tools).intersection(set(selected_tools)))
            result_single_test["accuracy"] = overlap / max(
                len(expected_tools), len(selected_tools)
            )
```
2. Test pass - Set to True if at least one of the tool expected were picked.

## Results
The results were measured by:
1. The avarage accuracy.
2. The ratio between the test that passed to all the tests.
3. The ratio between the test that had an accuracy score above 0.5 to all the tests.

* Baseline (Normal Agent with no Middleware)
```
Normal Agent Mean Accuracy: 67.91%

Normal Agent test pass to all tests ratio: 92.00%

Normal Agent above threshold accuracy ratio: 51.60%

Normal Agent Mean Total tokens: 24369.10

Normal Agent Mean Time per Prompt: 81.25 sec
```

* Embeddings Search:
```
Filtering Agent Mean Accuracy: 83.87%

Filtering Agent test pass to all tests ratio: 95.28%

Filtering Agent above threshold accuracy ratio: 74.34%
```

* BM25 Search (Filtering Agent)
```
Filtering Agent Mean Accuracy: 78.12%

Filtering Agent test pass to all tests ratio: 91.20%

Filtering Agent above threshold accuracy ratio: 68.80%

Filtering Agent Mean Total tokens: 7477.96

Filtering Agent Mean Time per Prompt: 82.21 sec
```

### Discussion
As we can see, the Filtering agent (AKA the agent that is using the BM25 search-based tools selection) can actually improve the accuracy of the agent when using tools.

1. About 10% Accuracy improvement and about 17% increase in the accuracy above threshold 0.5 means that the tool selection can work, and by improving the search technique, we might achieve better results.

2. Additional test that counted the total tokens shows dramatic decrease of ~69%.


## Restrictions 
* I run this program locally using LM-Studio LLM `meta-llama-3.1-8b-instruct-128k` on a M4 Mac.
* Due to hardware limitations, I used the first 50 tools and tested with the queries that included them. 

## High-Level Architecture
```
User Prompt
     ↓
Embedding-Based Tool Retrieval
     ↓
Top-K Candidate Tools
     ↓
Final Allowed Tool Set
     ↓
LangChain Agent (No Tool Execution)
     ↓
Evaluation Against Ground Truth
```

## Repo Structure
```
├── README.md
├── embeddings.py
├── prototype
│   ├── intialize_agent.py
│   ├── main.py
│   └── tools.py
├── pyproject.toml
├── test
│   ├── agent_tool_calling_init.py
│   ├── data
│   │   ├── test_suite.json
│   │   └── xlam_processed.json
│   ├── generate_data.py
│   ├── generated_tools.py
│   ├── post_processing.py
│   ├── results
│   │   ├── backup
│   ├── test_runner.py
│   └── tools_embeddings.py
└── uv.lock
```

## LangChain Integration

This project integrates with LangChain’s agent middleware, including:
* LLMToolSelectorMiddleware
* wrap_model_call for tool visibility restriction

Agents are restricted to the filtered tool set per request, enforcing correctness by construction.

## Requirements

* Python 3.11
* uv package manager

```
tools-selector v0.1.0
├── datasets v4.4.1
├── docx2txt v0.9
├── langchain v1.0.8
├── langchain-core v1.1.0
├── langchain-openai v1.0.3
├── llama-index v0.14.8
├── llama-index-embeddings-huggingface v0.6.1
├── llama-index-llms-openai v0.6.9
├── llama-index-llms-openai-like v0.5.3
├── loguru v0.7.3
├── numpy v2.3.5
├── openai v2.8.1
├── pandas v2.2.3
├── pydantic v2.12.4
├── torch v2.9.1
├── tqdm v4.67.1
└── transformers v4.57.1
```