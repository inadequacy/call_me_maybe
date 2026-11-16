*This project has been created as part of the 42 curriculum by the_inadequate.*

# Call Me Maybe

## Description

This project converts natural-language requests into function calls described by a JSON function schema. It uses the provided `Qwen/Qwen3-0.6B` model to select a function and extract its arguments. Token-by-token constrained decoding restricts generated tokens to prefixes that can form valid function names, JSON strings, numbers, integers, or booleans.

The output contains the original prompt, the selected function name, and a parameter object. The program does not execute the selected function.

## Instructions

Use Python 3.10 or newer and `uv`. The supplied `llm_sdk` wrapper uses PyTorch, Transformers, Hugging Face Hub, and Accelerate (for automatic GPU placement); these are runtime dependencies of that wrapper.

```sh
uv sync
uv run python -m src
```

Default input files are `data/input/functions_definition.json` and `data/input/function_calling_tests.json`. The default output is `data/output/function_calls.json`.

Custom paths and a different compatible model can be supplied as follows:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json \
  --model Qwen/Qwen3-0.6B
```

The first run needs access to the model files. `uv sync` resolves and installs the runtime dependencies before running the project. `make run`, `make debug`, `make lint`, and `make clean` provide common development commands.

## Algorithm

1. Load and validate the function definitions and prompts with Pydantic.
2. Build a model prompt containing the natural-language request and the available function names.
3. At each generation step, obtain the model's next-token logits. Keep only tokens whose decoded text remains a prefix of one of the JSON-encoded function names, set all other logits to negative infinity, then select the highest-scoring allowed token.
4. For every required parameter, prompt the model to extract its value. Apply a type-specific prefix rule: JSON string syntax for strings, a JSON-number grammar for numbers and integers, and the finite choices `true` or `false` for booleans.
5. Serialize the results as JSON, rejecting non-finite numbers and reporting input, inference, or output errors clearly.

This guarantees the lexical shape of each generated value. It does not guarantee that the model understood the request correctly; semantic accuracy still needs review against the prompt.

## Design decisions

- Function selection is made by the language model; there is no keyword-to-function routing table.
- The decoder relies on public SDK methods: `encode`, `get_logits_from_input_ids`, and `get_path_to_vocab_file`.
- Pydantic models reject malformed inputs and unsupported argument types before inference.
- Function arguments are emitted in the order declared by the schema, and only required parameter keys are written.
- Input and output paths are configurable to support alternate review data.

## Performance and reliability

The model is loaded once per run and reused across prompts. Decoded vocabulary pieces are cached. Each token step checks vocabulary candidates, so generation cost grows with the vocabulary size and output length; benchmark this on the evaluation hardware before claiming a specific runtime. JSON serialization and type-specific decoding constraints provide structural reliability, while function choice and extracted values remain dependent on model quality.

## Challenges

The main challenge is that asking a language model to “return JSON” does not enforce a schema. The decoder therefore masks next-token logits using a prefix rule. Free-form strings and numeric values need different prefix rules because their valid character sequences differ.

## Testing strategy

Before submission, run the program with the provided examples and verify that the output parses as JSON and that every function name, argument key, and argument type matches `functions_definition.json`. Also check empty and malformed input files, custom paths, quoted strings, negative and decimal numbers, and ambiguous prompts. Peer-review the semantic result for each prompt; valid JSON alone is not evidence of a correct function call.

## Example

For a request such as `What is the sum of 2 and 3?`, an expected call has this shape:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources and AI use

- The assignment brief, `call me maybe: Introduction to function calling in LLMs`.
- Python documentation: <https://docs.python.org/3/>
- Pydantic documentation: <https://docs.pydantic.dev/>
- NumPy documentation: <https://numpy.org/doc/>
- The provided `llm_sdk` package and its public API.

AI assistance was used to inspect the assignment, plan the decoder and CLI, draft implementation code, and prepare this README. The author should review and understand the generated code, run it against the project data, check semantic accuracy, and discuss the implementation with peers before submission.
