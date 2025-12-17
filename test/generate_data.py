"""
xlam-function-calling-60k Dataset Processing and Tool Generation

This module downloads the xlam-function-calling-60k dataset from HuggingFace,
extracts unique function definitions, generates dynamic @tool decorated functions,
and creates a comprehensive JSON test suite for evaluating tool selection.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set

from datasets import load_dataset


class XLAMDatasetProcessor:
    """Main class for processing xlam dataset and generating tools"""

    def __init__(self, cache_dir: str = "test/data"):
        self.cache_dir = cache_dir
        self.processed_data_path = os.path.join(cache_dir, "xlam_processed.json")
        self.test_suite_path = os.path.join(cache_dir, "test_suite.json")

        # Mock value templates by category
        self.mock_values = {
            "math": {
                "add": 42,
                "subtract": 15,
                "multiply": 156,
                "divide": 3.14,
                "sum": 12345,
                "average": 123.45,
                "max": 999,
                "min": 1,
                "default": 42
            },
            "text": {
                "uppercase": "SAMPLE TEXT",
                "lowercase": "sample text",
                "length": 25,
                "reverse": "txet elpmas",
                "split": ["sample", "text"],
                "join": "sample text",
                "replace": "sample text",
                "default": "Sample processed text"
            },
            "weather": {
                "default": "The weather in {location} is sunny and 22°C with light winds."
            },
            "stock": {
                "default": "The price of {symbol} is $156.78 USD, up 2.3% today."
            },
            "time": {
                "current_time": "2025-12-15 14:30:00 UTC",
                "format_time": "2025-12-15 14:30:00",
                "parse_time": "2025-12-15 14:30:00",
                "default": "2025-12-15 14:30:00 UTC"
            },
            "data": {
                "sort": [1, 2, 3, 4, 5],
                "filter": [1, 2, 3],
                "map": [2, 4, 6],
                "length": 5,
                "default": [1, 2, 3, 4, 5]
            },
            "file": {
                "read": "Content of file: {path}",
                "write": "Successfully wrote to {path}",
                "exists": True,
                "size": 1024,
                "default": "File operation completed"
            },
            "network": {
                "get": "Response from {url}",
                "post": "Posted data to {url}",
                "download": "Downloaded {url}",
                "default": "Network request completed"
            },
            "boolean": {
                "is_valid": True,
                "exists": True,
                "is_empty": False,
                "default": True
            },
            "default": "Mock function result"
        }

    def download_and_process_dataset(self) -> Dict[str, Any]:
        """Download xlam dataset and process it for tool extraction"""
        print("Downloading xlam-function-calling-60k dataset...")

        if os.path.exists(self.processed_data_path):
            print(f"Loading cached dataset from {self.processed_data_path}")
            with open(self.processed_data_path, 'r') as f:
                return json.load(f)

        try:
            # Download the dataset
            dataset = load_dataset("Salesforce/xlam-function-calling-60k")
            train_data = dataset['train']

            print(f"Dataset loaded: {len(train_data)} examples")

            processed_data = {
                "metadata": {
                    "dataset_name": "xlam-function-calling-60k",
                    "total_examples": len(train_data),
                    "processed_date": datetime.now().isoformat(),
                    "source": "Salesforce/xlam-function-calling-60k"
                },
                "examples": []
            }

            # Process each example
            for i, example in enumerate(train_data):
                if i % 1000 == 0:
                    print(f"Processing example {i}/{len(train_data)}")

                processed_example = {
                    "id": f"example_{i:05d}",
                    "query": example.get('query', ''),
                    "tools": self._parse_tools_from_example(example),
                    "answer": example.get('answer', ''),
                }
                processed_data["examples"].append(processed_example)

            # Save processed data
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            with open(self.processed_data_path, 'w') as f:
                json.dump(processed_data, f, indent=2)

            print(f"Processed data saved to {self.processed_data_path}")
            return processed_data

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            # Return empty structure for testing
            return {
                "metadata": {
                    "dataset_name": "xlam-function-calling-60k",
                    "total_examples": 0,
                    "processed_date": datetime.now().isoformat(),
                    "error": str(e)
                },
                "examples": []
            }

    def _parse_tools_from_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool definitions from a dataset example"""
        tools = []

        # Check if example has 'tools' field
        if 'tools' in example and example['tools']:
            if isinstance(example['tools'], str):
                try:
                    tools_data = json.loads(example['tools'])
                except json.JSONDecodeError:
                    return []
            else:
                tools_data = example['tools']

            if isinstance(tools_data, list):
                for tool_def in tools_data:
                    if isinstance(tool_def, dict):
                        tools.append({
                            "name": tool_def.get('name', 'unknown_function'),
                            "description": tool_def.get('description', ''),
                            "parameters": tool_def.get('parameters', {}),
                            "required": tool_def.get('required', []),
                            "type": tool_def.get('type', 'function')
                        })

        # Also check for function calls in the answer
        if 'answer' in example and example['answer']:
            function_calls = self._extract_function_calls_from_text(example['answer'])
            for func_name in function_calls:
                if not any(tool['name'] == func_name for tool in tools):
                    tools.append({
                        "name": func_name,
                        "description": f"Function {func_name} extracted from answer",
                        "parameters": {"type": "object", "properties": {}},
                        "required": [],
                        "type": "function"
                    })

        return tools

    def _extract_function_calls_from_text(self, text: str) -> Set[str]:
        """Extract function call names from text using regex"""
        function_pattern = r'(\w+)\s*\('
        matches = re.findall(function_pattern, text)

        # Filter out common words that aren't function names
        common_words = {'if', 'for', 'while', 'with', 'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'range', 'max', 'min', 'abs', 'sum'}
        function_names = {match for match in matches if match not in common_words and len(match) > 1}

        return function_names

    def extract_unique_tools(self, processed_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract unique tools from processed dataset"""
        print("Extracting unique tools from dataset...")

        unique_tools = {}
        tool_counts = defaultdict(int)

        for example in processed_data.get("examples", []):
            for tool in example.get("tools", []):
                tool_name = tool.get("name", "").strip()
                if not tool_name or tool_name == "unknown_function":
                    continue

                tool_counts[tool_name] += 1

                # Keep the most detailed version of each tool
                if tool_name not in unique_tools or len(tool.get("description", "")) > len(unique_tools[tool_name].get("description", "")):
                    unique_tools[tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", f"Function for {tool_name}"),
                        "parameters": tool.get("parameters", {}),
                        "required": tool.get("required", []),
                        "category": self._categorize_tool(tool_name, tool.get("description", "")),
                        "usage_count": tool_counts[tool_name]
                    }

        print(f"Extracted {len(unique_tools)} unique tools")
        return unique_tools

    def _categorize_tool(self, tool_name: str, description: str) -> str:
        """Categorize a tool based on its name and description"""
        name_lower = tool_name.lower()
        desc_lower = description.lower()

        # Math/calculation keywords
        math_keywords = ['add', 'subtract', 'multiply', 'divide', 'sum', 'average', 'mean', 'max', 'min', 'calculate', 'math', 'number', 'count']
        if any(keyword in name_lower or keyword in desc_lower for keyword in math_keywords):
            return "math"

        # Text processing keywords
        text_keywords = ['text', 'string', 'word', 'char', 'upper', 'lower', 'split', 'join', 'replace', 'format', 'parse']
        if any(keyword in name_lower or keyword in desc_lower for keyword in text_keywords):
            return "text"

        # Weather keywords
        weather_keywords = ['weather', 'temperature', 'forecast', 'climate', 'rain', 'snow', 'wind']
        if any(keyword in name_lower or keyword in desc_lower for keyword in weather_keywords):
            return "weather"

        # Stock/finance keywords
        stock_keywords = ['stock', 'price', 'finance', 'market', 'trading', 'investment', 'portfolio']
        if any(keyword in name_lower or keyword in desc_lower for keyword in stock_keywords):
            return "stock"

        # Time/date keywords
        time_keywords = ['time', 'date', 'timestamp', 'schedule', 'calendar', 'hour', 'minute', 'second']
        if any(keyword in name_lower or keyword in desc_lower for keyword in time_keywords):
            return "time"

        # Data processing keywords
        data_keywords = ['sort', 'filter', 'map', 'reduce', 'list', 'array', 'data', 'process']
        if any(keyword in name_lower or keyword in desc_lower for keyword in data_keywords):
            return "data"

        # File operations keywords
        file_keywords = ['file', 'read', 'write', 'save', 'load', 'path', 'directory', 'folder']
        if any(keyword in name_lower or keyword in desc_lower for keyword in file_keywords):
            return "file"

        # Network keywords
        network_keywords = ['http', 'url', 'api', 'request', 'download', 'upload', 'get', 'post', 'fetch']
        if any(keyword in name_lower or keyword in desc_lower for keyword in network_keywords):
            return "network"

        return "default"

    def generate_tool_functions(self, unique_tools: Dict[str, Dict[str, Any]]) -> str:
        """Generate Python code for @tool decorated functions"""
        print("Generating tool functions...")

        generated_code = '''"""
Dynamically generated tools from xlam-function-calling-60k dataset
Generated on: {timestamp}
Total tools: {tool_count}
"""

from typing import Any, Dict, List, Union
from langchain_core.tools import tool

# Generated tool functions
'''.format(
            timestamp=datetime.now().isoformat(),
            tool_count=len(unique_tools)
        )

        for tool_name, tool_data in unique_tools.items():
            # Generate function signature
            parameters = tool_data.get("parameters", {}).get("properties", {})
            required_params = tool_data.get("required", [])

            # Create function parameters
            params = []
            for param_name, param_info in parameters.items():
                if isinstance(param_info, dict):
                    param_type = self._python_type_from_json_schema(param_info.get("type", "any"))
                else:
                    param_type = "Any"

                if param_name not in required_params:
                    params.append(f"{param_name}: {param_type} = None")
                else:
                    params.append(f"{param_name}: {param_type}")

            if not params:
                params = ["*args", "**kwargs"]

            param_str = ", ".join(params)

            # Get mock return value
            category = tool_data.get("category", "default")
            mock_value = self._get_mock_value(tool_name, category, parameters)

            # Generate function code
            function_code = f'''
@tool
def {tool_name}({param_str}) -> Any:
    """{tool_data.get("description", f"Function for {tool_name}")}"""
    print(f"\\n{tool_name.replace('_', ' ').title()}\\n")
    return {repr(mock_value)}
'''
            generated_code += function_code

        # Add tool array function
        tool_names = list(unique_tools.keys())
        generated_code += f'''

# Tool array for LangChain integration
def get_generated_tools():
    """Return array of all generated tools"""
    return [{", ".join(tool_names)}]

def get_generated_tool_names():
    """Return list of tool names"""
    return {repr(tool_names)}
'''

        return generated_code

    def _python_type_from_json_schema(self, json_type: str) -> str:
        """Convert JSON schema type to Python type hint"""
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
            "null": "None"
        }
        return type_mapping.get(json_type, "Any")

    def _get_mock_value(self, tool_name: str, category: str, parameters: Dict[str, Any]) -> Any:
        """Generate appropriate mock return value for a tool"""
        name_lower = tool_name.lower()

        # Category-specific mock values
        if category in self.mock_values:
            category_mocks = self.mock_values[category]

            # Handle string values (like single default values)
            if isinstance(category_mocks, str):
                return category_mocks

            # Handle dictionary values
            if isinstance(category_mocks, dict):
                # Try to find specific mock for this function name
                for key, value in category_mocks.items():
                    if key in name_lower:
                        return value

                # Use category default
                if "default" in category_mocks:
                    return category_mocks["default"]

        # Fallback based on function name patterns
        if any(word in name_lower for word in ["is_", "has_", "can_", "should_", "check"]):
            return True
        elif any(word in name_lower for word in ["count", "length", "size", "num"]):
            return 42
        elif any(word in name_lower for word in ["list", "array", "items"]):
            return [1, 2, 3, 4, 5]
        elif any(word in name_lower for word in ["dict", "map", "object"]):
            return {"result": "success", "value": 42}

        return self.mock_values["default"]

    def build_test_cases(self, processed_data: Dict[str, Any], unique_tools: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build test cases from processed data for evaluation"""
        print("Building test cases...")

        test_suite = {
            "metadata": {
                "dataset_name": "xlam-function-calling-60k",
                "generated_date": datetime.now().isoformat(),
                "total_tools": len(unique_tools),
                "total_test_cases": 0
            },
            "tools": list(unique_tools.values()),
            "test_cases": [],
            "evaluation_metrics": {
                "tool_selection_accuracy": "percentage",
                "parameter_extraction_accuracy": "percentage",
                "execution_success_rate": "percentage"
            }
        }

        # Create test cases from examples
        for example in processed_data.get("examples", []):
            if not example.get("query") or not example.get("tools"):
                continue

            expected_tools = [tool["name"] for tool in example["tools"] if tool["name"] in unique_tools]
            if not expected_tools:
                continue

            # Categorize difficulty
            difficulty = self._categorize_difficulty(example["query"], expected_tools)

            test_case = {
                "id": example["id"],
                "query": example["query"],
                "expected_tools": expected_tools,
                "expected_calls": [
                    {
                        "tool": tool["name"],
                        "parameters": tool.get("parameters", {})
                    }
                    for tool in example["tools"] if tool["name"] in unique_tools
                ],
                "difficulty": difficulty,
                "category": self._get_dominant_category(expected_tools, unique_tools)
            }
            test_suite["test_cases"].append(test_case)

        test_suite["metadata"]["total_test_cases"] = len(test_suite["test_cases"])
        return test_suite

    def _categorize_difficulty(self, query: str, expected_tools: List[str]) -> str:
        """Categorize the difficulty of a test case"""
        if len(expected_tools) <= 1:
            return "easy"
        elif len(expected_tools) <= 3:
            return "medium"
        else:
            return "hard"

    def _get_dominant_category(self, tool_names: List[str], unique_tools: Dict[str, Dict[str, Any]]) -> str:
        """Get the dominant category for a set of tools"""
        categories = defaultdict(int)
        for tool_name in tool_names:
            if tool_name in unique_tools:
                category = unique_tools[tool_name].get("category", "default")
                categories[category] += 1

        if not categories:
            return "default"

        return max(categories.items(), key=lambda x: x[1])[0]

    def export_to_json(self, test_suite: Dict[str, Any]) -> str:
        """Export test suite to JSON file"""
        print("Exporting test suite to JSON...")

        with open(self.test_suite_path, 'w') as f:
            json.dump(test_suite, f, indent=2)

        print(f"Test suite exported to {self.test_suite_path}")
        return self.test_suite_path


def main():
    """Main function to run the dataset processing and tool generation"""
    print("Starting xlam dataset processing...")

    processor = XLAMDatasetProcessor()

    # Step 1: Download and process dataset
    processed_data = processor.download_and_process_dataset()

    # Step 2: Extract unique tools
    unique_tools = processor.extract_unique_tools(processed_data)

    # Step 3: Generate tool functions
    tool_code = processor.generate_tool_functions(unique_tools)

    # Save generated tools
    generated_tools_path = "test/generated_tools.py"
    with open(generated_tools_path, 'w') as f:
        f.write(tool_code)

    print(f"Generated tools saved to {generated_tools_path}")
    print(f"Total unique tools: {len(unique_tools)}")

    # Step 4: Build test cases
    test_suite = processor.build_test_cases(processed_data, unique_tools)

    # Step 5: Export to JSON
    processor.export_to_json(test_suite)

    # Display some statistics
    categories = defaultdict(int)
    for tool_data in unique_tools.values():
        categories[tool_data.get("category", "default")] += 1

    print("\nTool categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    difficulty_counts = defaultdict(int)
    for test_case in test_suite["test_cases"]:
        difficulty_counts[test_case["difficulty"]] += 1

    print("\nTest case difficulties:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")


if __name__ == "__main__":
    main()