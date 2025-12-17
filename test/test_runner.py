import asyncio
import json
import os
import sys
from time import time
from typing import List

from agent_tool_calling_init import (
    _NUM_TOOLS,
    _TOOLS_EMBEDDINGS,
    initialize_agent,
    initialize_filtered_agent,
)
from loguru import logger
from tqdm.asyncio import tqdm

logger.remove()

logger.add(sys.stderr, level="DEBUG")

_INVOKE_TIMEOUT: int = 360

class ToolSelectionEvaluator:
    """Evaluates tool selection performance using xlam test cases"""

    def __init__(
        self,
        agent_type: str,
        max_tools: int,
        test_suite_path: str = "data/test_suite.json",
        results_path="results",
    ):
        """
        agent_type: normal or filtering
        """
        assert agent_type is not None and agent_type != "", "Must provide an agent for this test!"
        assert max_tools == -1 or max_tools > 0, "Either use some tools or all tools"
        self.max_tools = max_tools

        if agent_type == "normal":
            self.agent = initialize_agent(max_tools=self.max_tools)
        elif agent_type == "filtering":
            self.agent = initialize_filtered_agent(max_tools=self.max_tools)
        else:
            raise ValueError("Agent is 'normal' of 'filtering' ")

        self.test_suit_path = test_suite_path
        self.test_cases = None
        self.results_file_path = results_path + f"/{agent_type}_agent_results.json"

        self.results_summary = {
            "total_test_time": -1,
            "total_test": 0,
            "total_failed": 0,
            "total_passed": 0,
            "tool_selection_accuracy": 0.0,
        }

    def _load_eval_data(self) -> bool:
        try:
            with open(self.test_suit_path, "r") as file:
                self.test_cases = json.load(file)
            logger.info(f"Loaded test file with {len(self.test_cases['test_cases'])} test cases")
            return True
        except Exception as e:
            logger.error(f"Could not load file {self.test_suit_path} due to: {e}")
            return False

    def _extract_tool_calls(self, response):
        assert "messages" in response.keys(), (
            f"Response has {response.keys()}, no 'messages' found."
        )

        tool_calls = []

        for message in response["messages"]:
            if hasattr(message, "tool_calls"):
                tool_call = getattr(message, "tool_calls")
                for tc in tool_call:
                    if "name" in tc:
                        tool_calls.append(tc["name"])

        return list(set(tool_calls))  # avoid duplications

    def _eval_single_case(self, test_case, n):
        result_single_test = {
            "test_id": test_case["id"],
            "total_time": -1,
            "used_tools": [],  # the tools that the agent actually used.
            "test_tools": [],  # the tools that the agent need to use.
            "test_pass": False,
            "accuracy": 0,
            "difficulty": test_case["difficulty"],
        }

        query = test_case["query"]
        expected_tools = set(test_case["expected_tools"])

        try:
            start_time = time()
            response = self.agent.invoke({"messages": [("user", query)]})
            end_time = time()
        except Exception as e:
            logger.error(f"Error: {e}\nOccured in test #{n} with id: {test_case['id']}")
            return result_single_test

        selected_tools = self._extract_tool_calls(response)

        result_single_test["total_time"] = end_time - start_time
        result_single_test["test_tools"] = list(expected_tools)
        result_single_test["used_tools"] = selected_tools

        if not expected_tools and not selected_tools:
            result_single_test["test_pass"] = True
            result_single_test["accuracy"] = 1.0
        elif expected_tools:
            overlap = len(set(expected_tools).intersection(set(selected_tools)))
            result_single_test["accuracy"] = overlap / max(
                len(expected_tools), len(selected_tools)
            )
            if result_single_test["accuracy"] > 0:
                result_single_test["test_pass"] = True
        return result_single_test

    def update_resuls(self, new_data: List):
        if os.path.exists(self.results_file_path):
            with open(self.results_file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        # Update with new data
        data.extend(new_data)

        # Write back
        with open(self.results_file_path, "w") as f:
            json.dump(data, f, indent=2)

    def run_evaluation(self):
        if not self._load_eval_data():
            raise ValueError("Could not open test file")

        counter = 0
        results = []
        all_tools = list(_TOOLS_EMBEDDINGS.keys())

        all_cases = self.test_cases["test_cases"]

        for i, test_case in enumerate(all_cases):
            if i % 10 == 0:
                logger.debug(f"Processing test #{i} out of {len(all_cases)} tests")
                if len(results) > 0:  # TODO: remove this
                    logger.info(f"dumping {len(results)} into file:")
                    self.update_resuls(new_data=results)
                    results = []

            if (
                len(set(test_case["expected_tools"]).intersection(set(all_tools)))
                != len(set(test_case["expected_tools"]))
                and len(test_case["expected_tools"]) != 0
            ):
                logger.debug(f"Skipping test #{i} due to tool mismatch...")
                continue

            logger.debug(
                f"testing case #{i} so far tested {counter} cases and collected {len(results)} results"
            )
            curr_result = self._eval_single_case(test_case, i)
            counter += 1
            if curr_result["total_time"] == -1:
                logger.warning(f"test #{i} failed.")
                continue  # test failed

            results.append(curr_result)

        if len(results) > 0:
            self.update_resuls(results)
            results = []

    #### Async tests ####

    async def _eval_case_async(self, test_case, n, semaphore: asyncio.Semaphore):
        """Async version of _eval_single_case"""
        async with semaphore:
            result = {
                "test_id": test_case["id"],
                "total_time": -1,
                "used_tools": [],
                "test_tools": [],
                "test_pass": False,
                "accuracy": 0,
                "difficulty": test_case["difficulty"],
            }

            query = test_case["query"]
            expected_tools = set(test_case["expected_tools"])

            try:
                start_time = time()
                response = await asyncio.wait_for(
                    self.agent.ainvoke({"messages": [("user", query)]}),
                    timeout=_INVOKE_TIMEOUT)
                end_time = time()
            except asyncio.TimeoutError:
                logger.error(f"Timeout Error occured in test #{n} with id: {test_case['id']}")
                return result
            except Exception as e:
                logger.error(f"Error: {e}\nOccured in test #{n} with id: {test_case['id']}")
                return result

            selected_tools = self._extract_tool_calls(response)

            result["total_time"] = end_time - start_time
            result["test_tools"] = list(expected_tools)
            result["used_tools"] = selected_tools

            if not expected_tools and not selected_tools:
                result["test_pass"] = True
                result["accuracy"] = 1.0
            elif expected_tools:
                overlap = len(expected_tools.intersection(set(selected_tools)))
                result["accuracy"] = overlap / max(len(expected_tools), len(selected_tools))
                if result["accuracy"] > 0:
                    result["test_pass"] = True

            return result
    async def run_parallel_evaluation(self, max_concurrent: int = 5):
        """Run evaluation with parallel async calls"""
        if not self._load_eval_data():
            raise ValueError("Could not open test file")

        semaphore = asyncio.Semaphore(max_concurrent)
        all_tools = list(_TOOLS_EMBEDDINGS.keys())
        all_cases = self.test_cases["test_cases"]

        # Filter valid test cases
        valid_cases = [
            (i, tc) for i, tc in enumerate(all_cases)
            if len(set(tc["expected_tools"]).intersection(set(all_tools))) == len(set(tc["expected_tools"]))
            or len(tc["expected_tools"]) == 0
        ]

        logger.info(f"Running {len(valid_cases)} test cases with max {max_concurrent} concurrent")

        # Create all tasks
        tasks = [
            self._eval_case_async(tc, i, semaphore)
            for i, tc in valid_cases
        ]

        # Run in parallel
        valid_results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            result = await coro
            if not isinstance(result, Exception) and result["total_time"] != -1:
                valid_results.append(result)


        logger.info(f"Completed {len(valid_results)} successful tests")
        if len(valid_results) > 0:
            self.update_resuls(valid_results)
 


if __name__ == "__main__":
    # print("TEST STARTS")
    # evaluator = ToolSelectionEvaluator(agent_type="normal", max_tools=_NUM_TOOLS)
    # evaluator.run_evaluation()
    # print("TEST ENDS")

    print("TEST STARTS")
    evaluator = ToolSelectionEvaluator(agent_type="filtering", max_tools=_NUM_TOOLS)
    asyncio.run(evaluator.run_parallel_evaluation(max_concurrent=4))
    print("TEST ENDS")
