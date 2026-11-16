"""Command-line function-calling application."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.schema import FunctionDefinition, PromptInput


def _read_json(path: Path) -> object:
    """Read JSON from a path with a useful error message."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except FileNotFoundError as error:
        raise ValueError(f"Input file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line paths and model option."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert natural-language requests to constrained function calls."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/output/function_calls.json")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    return parser.parse_args(argv)


def run(
    functions_path: Path,
    input_path: Path,
    output_path: Path,
    model_name: str = "Qwen/Qwen3-0.6B",
) -> None:
    """Process all prompts and write schema-valid function-call JSON."""
    raw_functions = _read_json(functions_path)
    raw_prompts = _read_json(input_path)
    if not isinstance(raw_functions, list) or not isinstance(
        raw_prompts, list
    ):
        raise ValueError(
            "Both function definitions and prompts must be JSON arrays."
        )
    try:
        functions = [
            FunctionDefinition.model_validate(item) for item in raw_functions
        ]
        prompts = [PromptInput.model_validate(item) for item in raw_prompts]
    except ValidationError as error:
        raise ValueError(
            f"Input does not match the required schema: {error}"
        ) from error
    if not functions:
        raise ValueError("At least one function definition is required.")
    if len({function.name for function in functions}) != len(functions):
        raise ValueError("Function names must be unique.")

    try:
        from llm_sdk import Small_LLM_Model
        from src.decoder import ConstrainedDecoder
    except ImportError as error:
        raise ValueError(
            f"The model SDK dependencies are unavailable: {error}"
        ) from error

    model = Small_LLM_Model(model_name)
    decoder = ConstrainedDecoder(model)
    by_name = {function.name: function for function in functions}
    results: list[dict[str, object]] = []
    for item in prompts:
        selected_name = decoder.choose_function(item.prompt, list(by_name))
        selected = by_name[selected_name]
        parameters = {
            name: decoder.generate_argument(
                item.prompt, selected.name, name, definition.type
            )
            for name, definition in selected.parameters.items()
        }
        selected.validate_arguments(parameters)
        results.append(
            {
                "prompt": item.prompt,
                "name": selected.name,
                "parameters": parameters,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                results,
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            output_file.write("\n")
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not write output to {output_path}: {error}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application and report expected input or inference errors."""
    args = _parse_args(argv)
    try:
        run(args.functions_definition, args.input, args.output, args.model)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
