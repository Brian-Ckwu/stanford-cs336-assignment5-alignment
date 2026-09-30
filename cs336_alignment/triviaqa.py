"""TriviaQA string matching, following llm-calibration's dataset verifier."""

import re
from typing import Any, Mapping


def extract_answer(response: str) -> str | None:
    response = response.strip()
    for pattern in (
        r"\\boxed\{([^}]+)\}",
    ):
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return re.sub(r"[.!?,;:]+$", "", matches[-1].strip()).strip()
    return None


def normalize_answer(text: str) -> str:
    text = re.sub(r"\b(the|a|an)\b", "", text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def triviaqa_reward_fn(response: str, ground_truth: Mapping[str, Any]) -> dict[str, float]:
    extracted = extract_answer(response)
    prediction = normalize_answer(extracted) if extracted is not None else ""
    correct = False
    if prediction:
        for reference in [ground_truth["value"], *ground_truth.get("aliases", [])]:
            reference = normalize_answer(reference)
            if reference and (prediction in reference or reference in prediction):
                correct = True
                break
        if not correct:
            correct = any(
                prediction == normalize_answer(alias)
                for alias in ground_truth.get("normalized_aliases", [])
            )
    return {
        "format_reward": float(bool(prediction)),
        "answer_reward": float(correct),
        "reward": float(correct),
    }
