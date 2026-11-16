"""Validated data models for the function-calling inputs and outputs."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ArgumentDefinition(BaseModel):
    """Describe one supported JSON argument type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """Describe a callable function and its required arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ArgumentDefinition]
    returns: dict[str, str]

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Reject missing, extra, or incorrectly typed argument values."""
        if set(arguments) != set(self.parameters):
            raise ValueError(
                "Generated parameter names do not match the function schema."
            )
        for name, definition in self.parameters.items():
            value = arguments[name]
            valid = {
                "string": lambda item: isinstance(item, str),
                "number": lambda item: (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                ),
                "integer": lambda item: isinstance(item, int)
                and not isinstance(item, bool),
                "boolean": lambda item: isinstance(item, bool),
            }[definition.type](value)
            if not valid:
                raise ValueError(
                    f"Generated argument {name!r} has the wrong type."
                )


class PromptInput(BaseModel):
    """Represent one natural-language request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCall(BaseModel):
    """Represent a result in the required output format."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
