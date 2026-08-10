"""Prompt rendering and confidence parsing shared by CC training and validation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal


ConfidenceOutputFormat = Literal["answer_tags", "boxed"]
ParseOutputFormat = Literal["auto", "answer_tags", "boxed"]
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def configure_chat_template(
    tokenizer: Any,
    *,
    use_chat_template: bool,
    chat_template_path: str | Path | None = None,
) -> None:
    """Configure and validate the tokenizer chat template used for CC prompts."""
    if chat_template_path is not None and not use_chat_template:
        raise ValueError("--chat-template-path requires --use-chat-template.")
    if not use_chat_template:
        return

    if chat_template_path is not None:
        path = Path(chat_template_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chat template does not exist: {path}")
        tokenizer.chat_template = path.read_text()
    elif not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer does not provide a chat template. Supply one with "
            "--chat-template-path."
        )

    try:
        render_user_prompt(
            tokenizer,
            "Validate the configured chat template.",
            use_chat_template=True,
        )
    except Exception as error:
        raise ValueError(f"Could not render the configured chat template: {error}") from error


def render_user_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    use_chat_template: bool,
) -> str:
    """Render one user message, including the assistant generation prefix."""
    if not use_chat_template:
        return user_prompt
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return str.")
    return rendered


def _parse_numeric_text(text: str) -> tuple[float | None, str | None]:
    try:
        value = float(text.strip())
    except ValueError:
        return None, "answer_not_number"
    if not math.isfinite(value):
        return None, "answer_not_finite"
    if not 0.0 <= value <= 1.0:
        return None, "answer_out_of_range"
    return value, None


def parse_confidence(
    response: str,
    *,
    output_format: ParseOutputFormat = "auto",
) -> tuple[float | None, str | None]:
    """Parse a confidence using the selected training/output contract."""
    if output_format == "auto":
        match = re.search(
            r"<answer>\s*(.*?)\s*(?:</answer>|$)",
            response.strip(),
            flags=re.DOTALL,
        )
        if match is None:
            return parse_confidence(response, output_format="boxed")
        answer_text = match.group(1).strip()
        if r"\boxed" in answer_text:
            return parse_confidence(answer_text, output_format="boxed")
        parsed, error = _parse_numeric_text(answer_text)
        if error != "answer_not_number":
            return parsed, error
        numbers = _NUMBER_RE.findall(answer_text)
        if len(numbers) != 1:
            return None, "answer_not_single_number"
        return _parse_numeric_text(numbers[0])

    if output_format == "answer_tags":
        match = re.search(
            r"</think>\s*<answer>\s*(.*?)\s*</answer>",
            response,
            flags=re.DOTALL,
        )
        if match is None:
            return None, "missing_answer_tags"
        answer_text = match.group(1)
        if r"\boxed" in answer_text:
            return parse_confidence(answer_text, output_format="boxed")
        return _parse_numeric_text(answer_text)

    if output_format == "boxed":
        matches = re.findall(r"\\boxed\{([0-9.]+)\}", response)
        if not matches:
            return None, "missing_boxed_answer"
        parsed_values = []
        for match in matches:
            try:
                parsed_values.append(float(match))
            except ValueError:
                continue
        for value in reversed(parsed_values):
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                return value, None
        return None, "boxed_answer_out_of_range"

    raise ValueError(f"Unknown confidence output format: {output_format}")
