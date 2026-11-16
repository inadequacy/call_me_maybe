"""Token-by-token constrained decoding helpers."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


PrefixRule = Callable[[str], bool]


class ConstrainedDecoder:
    """Generate text while masking tokens that violate a prefix rule.

    Vocabulary entries are read from the public SDK vocabulary path. The
    vocabulary token spellings are the pieces used by the constrained prefix
    rules, keeping the per-token masking loop local and inexpensive.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """Initialize the decoder and load the model vocabulary."""
        self.model = model
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> dict[int, str]:
        """Load IDs from a standard token-to-ID or ID-to-token JSON file."""
        path = self.model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as vocab_file:
            raw = json.load(vocab_file)
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                "The model vocabulary must be a non-empty JSON object."
            )
        sample = next(iter(raw.values()))
        if isinstance(sample, int):
            return {
                int(token_id): str(token) for token, token_id in raw.items()
            }
        return {int(token_id): str(token) for token_id, token in raw.items()}

    def _token_piece(self, token_id: int) -> str:
        """Return the vocabulary spelling for one token ID."""
        return self.vocab.get(token_id, "")

    def generate(
        self,
        prompt_text: str,
        prefix_is_valid: PrefixRule,
        is_complete: PrefixRule,
        max_tokens: int = 96,
    ) -> str:
        """Generate a value while restricting next tokens by a prefix rule.

        Args:
            prompt_text: Context supplied to the language model.
            prefix_is_valid: Returns true when text remains a valid prefix.
            is_complete: Returns true when text is a complete value.
            max_tokens: Maximum generated tokens.

        Returns:
            The constrained generated text.
        """
        encoded = self.model.encode(prompt_text)
        input_ids = (
            encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        )
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = [int(token_id) for token_id in input_ids]
        generated = ""

        for _ in range(max_tokens):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(input_ids), dtype=float
            )
            allowed: list[int] = []
            for token_id in self.vocab:
                if token_id >= len(logits):
                    continue
                piece = self._token_piece(token_id)
                if piece and prefix_is_valid(generated + piece):
                    allowed.append(token_id)
            if not allowed:
                break
            masked = np.full_like(logits, -np.inf)
            masked[allowed] = logits[allowed]
            next_id = int(np.argmax(masked))
            if not math.isfinite(float(masked[next_id])):
                break
            generated += self._token_piece(next_id)
            input_ids.append(next_id)
            if is_complete(generated):
                break
        if not is_complete(generated):
            raise ValueError(
                "The model could not complete a schema-valid value."
            )
        return generated

    def choose_function(
        self, prompt: str, function_names: Sequence[str]
    ) -> str:
        """Choose one function name from the provided names using the model."""
        choices = [
            json.dumps(name, ensure_ascii=False) for name in function_names
        ]

        def allowed_prefix(text: str) -> bool:
            return any(choice.startswith(text) for choice in choices)

        result = self.generate(
            "Choose the best function for this request. Return only one "
            "function name as a JSON string.\nAvailable functions:\n"
            + "\n".join(f"- {name}" for name in function_names)
            + f"\nRequest: {prompt}\nFunction name:",
            allowed_prefix,
            lambda text: text in choices,
            max_tokens=48,
        )
        return str(json.loads(result))

    def generate_argument(
        self, prompt: str, function_name: str, argument_name: str, kind: str
    ) -> object:
        """Generate one argument as a schema-constrained JSON primitive."""
        task = (
            f"Extract argument {argument_name!r} for "
            f"function {function_name!r} from this request: {prompt}\n"
            f"Return exactly one JSON {kind} value:"
        )
        if kind == "string":
            return json.loads(
                self.generate(task, _json_string_prefix, _json_string_complete)
            )
        if kind in ("number", "integer"):
            rule = _integer_prefix if kind == "integer" else _number_prefix
            raw = self.generate(
                task, rule, lambda value: _number_complete(value, kind)
            )
            number = json.loads(raw)
            if not isinstance(number, (int, float)) or isinstance(
                number, bool
            ):
                raise ValueError(
                    f"Generated value for {argument_name!r} was not numeric."
                )
            return number
        if kind == "boolean":
            choices = ("true", "false")
            raw = self.generate(
                task,
                lambda value: any(
                    choice.startswith(value) for choice in choices
                ),
                lambda value: value in choices,
                max_tokens=8,
            )
            return raw == "true"
        raise ValueError(f"Unsupported argument type: {kind}")


_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)
_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)")


def _number_prefix(value: str) -> bool:
    """Check whether text can still become a JSON number."""
    return value in ("", "-") or _NUMBER_PATTERN.fullmatch(value) is not None


def _integer_prefix(value: str) -> bool:
    """Check whether text can still become a JSON integer."""
    return value in ("", "-") or _INTEGER_PATTERN.fullmatch(value) is not None


def _number_complete(value: str, kind: str) -> bool:
    """Check whether text is a complete JSON number of the requested kind."""
    pattern = _INTEGER_PATTERN
    if kind != "integer":
        pattern = re.compile(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        )
    return pattern.fullmatch(value) is not None


def _json_string_prefix(value: str) -> bool:
    """Check that text is a prefix of a valid JSON string token."""
    if not value:
        return True
    if value[0] != '"':
        return False
    escaped = False
    for char in value[1:]:
        if escaped:
            if char not in '"\\/bfnrtu0123456789abcdefABCDEF':
                return False
            escaped = char == "u"
        elif char == "\\":
            escaped = True
        elif char == '"':
            return value.endswith('"')
        elif ord(char) < 0x20:
            return False
    return True


def _json_string_complete(value: str) -> bool:
    """Check whether text is exactly one valid JSON string."""
    if (
        not _json_string_prefix(value)
        or len(value) < 2
        or not value.endswith('"')
    ):
        return False
    try:
        return isinstance(json.loads(value), str)
    except json.JSONDecodeError:
        return False
