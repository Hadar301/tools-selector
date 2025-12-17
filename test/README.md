# Test Suite for Tool Selection Evaluation

This directory contains the evaluation framework for comparing tool selection strategies in LLM-based agents.

## Overview

The test suite evaluates two agent configurations:
- **Normal Agent**: Uses all available tools without filtering
- **Filtering Agent**: Uses embedding-based semantic search to pre-filter relevant tools

## Prerequisites

### 1. Python Environment

Requires Python 3.11 or higher.

### 2. LM Studio Setup

This project uses a local LLM served via [LM Studio](https://lmstudio.ai/).

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download the `meta-llama-3.1-8b-instruct-128k` model
3. Start the local server (default: `http://localhost:1234/`)
4. Verify the server is running before executing tests

> **Note**: The test runner will fail if LM Studio is not running.

## Installation

From the project root directory:

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Directory Structure

```
test/
├── README.md                  # This file
├── agent_tool_calling_init.py # Agent initialization and middleware
├── generate_data.py           # Dataset processing and tool generation
├── generated_tools.py         # Auto-generated tools from xLAM dataset
├── post_processing.py         # Results analysis utilities
├── test_runner.py             # Main test execution script
├── tools_embeddings.py        # Tool embedding utilities
├── data/
│   ├── xlam_processed.json    # Processed dataset (generated)
│   └── test_suite.json        # Test cases (generated)
└── results/
    ├── normal_agent_results.json    # Normal agent results
    └── filtering_agent_results.json # Filtering agent results
```

## Usage

### Step 1: Generate Test Data

First, download and process the xLAM dataset:

```bash
cd test
uv run generate_data.py
```

This will:
- Download the `xlam-function-calling-60k` dataset from HuggingFace
- Extract unique tool definitions
- Generate `generated_tools.py` with LangChain tool decorators
- Create `data/test_suite.json` with test cases

### Step 2: Run Evaluations

**Run the filtering agent evaluation (default):**

```bash
uv run test_runner.py
```

**To run the normal agent evaluation**, edit `test_runner.py` and change the `agent_type`:

```python
# In __main__ block:
evaluator = ToolSelectionEvaluator(agent_type="normal", max_tools=_NUM_TOOLS)
```

### Step 3: Analyze Results

After running evaluations, analyze the results:

```bash
uv run post_processing.py
```

This outputs:
- Mean accuracy for each agent
- Test pass ratio
- Accuracy above threshold (0.5) ratio

## Configuration

Key parameters in `agent_tool_calling_init.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `_NUM_TOOLS` | 50 | Number of tools to use from the dataset |
| `_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `_LLM_MODEL` | `meta-llama-3.1-8b-instruct-128k` | Model name |
| `threshold` | 0.25 | Similarity threshold for tool filtering |

Parameters in `test_runner.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `_INVOKE_TIMEOUT` | 360 | Timeout per test case (seconds) |
| `max_concurrent` | 4 | Parallel async test execution limit |

## Metrics

The evaluation tracks:

- **Accuracy**: Overlap between expected and selected tools divided by the maximum count
- **Test Pass**: True if at least one expected tool was selected
- **Difficulty**: easy (1 tool), medium (2-3 tools), hard (4+ tools)

## Troubleshooting

### Connection Error
```
AssertionError: http://localhost:1234/v1 appears like it is not active
```
**Solution**: Start LM Studio and load the model before running tests.

### Missing Data Files
```
Could not open test file
```
**Solution**: Run `python generate_data.py` first to generate the dataset.

### Timeout Errors
If tests timeout frequently, increase `_INVOKE_TIMEOUT` in `test_runner.py` or reduce `max_concurrent` for parallel runs.

