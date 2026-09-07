import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase

try:
    from .vllm_utils import VLLMServer
except ImportError:  # Support direct execution from cs336_alignment/.
    from vllm_utils import VLLMServer

def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    try:
        return str(completion.text)
    except AttributeError as error:
        raise TypeError("Completions must be strings or objects with a 'text' attribute.") from error


def _rank(values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[sorted_indices[end + 1]] == values[sorted_indices[start]]:
            end += 1
        ranks[sorted_indices[start:end + 1]] = (start + end) / 2
        start = end + 1
    return ranks


def _spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_ranks = _rank(np.asarray(x, dtype=float))
    y_ranks = _rank(np.asarray(y, dtype=float))
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    x_squared_deviations = float((x_centered**2).sum())
    y_squared_deviations = float((y_centered**2).sum())
    if x_squared_deviations == 0:
        return 0.0
    if y_squared_deviations == 0:
        return float("nan")
    denominator = math.sqrt(x_squared_deviations * y_squared_deviations)
    return float((x_centered * y_centered).sum() / denominator)


def _cc_token_id_to_count(
    tokenizer: PreTrainedTokenizerBase,
    group_size: int,
) -> dict[int, int]:
    if group_size <= 0:
        raise ValueError("cc_group_size must be positive.")
    token_id_to_count = {}
    for count in range(group_size + 1):
        token_ids = tokenizer.encode(str(count), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Expected confidence count {count} to encode to one token, got {token_ids}"
            )
        token_id = token_ids[0]
        if token_id in token_id_to_count:
            raise ValueError(
                f"Confidence counts {token_id_to_count[token_id]} and {count} map to the same token ID {token_id}"
            )
        token_id_to_count[token_id] = count
    return token_id_to_count


def evaluate(
    vllm_server: VLLMServer,
    eval_dataset: Sequence[Mapping[str, Any]],
    sampling_params: dict,
    batch_size: int,
    reward_fn: Callable,
    tokenizer: PreTrainedTokenizerBase | None = None,
    cc_group_size: int | None = None,
):
    print(f"Evaluating {len(eval_dataset)} instances ...")
    rows = list(eval_dataset)
    if not rows:
        raise ValueError(
            "valid_dataset is empty; check the configured validation dataset path "
            "and train/validation split sizes."
        )
    num_samples = sampling_params.get("n")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("Evaluation requires sampling_params['n'] to be a positive integer.")
    if (tokenizer is None) != (cc_group_size is None):
        raise ValueError("tokenizer and cc_group_size must either both be provided or both be None.")
    prompts = [row["prompt"] for row in rows]
    ground_truths = [row["answer"] for row in rows]
    completions = vllm_server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    responses = [_completion_text(completion) for completion in completions]
    repeated_ground_truths = [
        ground_truth
        for ground_truth in ground_truths
        for _ in range(num_samples)
    ]
    if len(responses) != len(repeated_ground_truths):
        raise ValueError(
            f"Expected {len(repeated_ground_truths)} responses for {len(prompts)} prompts "
            f"with n={num_samples}, got {len(responses)}"
        )
    eval_metrics = None
    answer_rewards = []
    for response, ground_truth in zip(responses, repeated_ground_truths):
        metrics = reward_fn(response=response, ground_truth=ground_truth)
        if cc_group_size is not None:
            if "answer_reward" not in metrics:
                raise KeyError("reward_fn must return answer_reward when CC evaluation is enabled.")
            answer_reward = float(metrics["answer_reward"])
            if not math.isfinite(answer_reward) or not 0.0 <= answer_reward <= 1.0:
                raise ValueError(f"answer_reward must be finite and in [0, 1], got {answer_reward}")
            answer_rewards.append(answer_reward)
        if eval_metrics is None:
            eval_metrics = dict(metrics)
        else:
            for k, v in metrics.items():
                eval_metrics[k] += v
    # Average the metrics
    for k in eval_metrics.keys():
        eval_metrics[k] /= len(responses)

    if cc_group_size is not None:
        assert tokenizer is not None
        token_id_to_count = _cc_token_id_to_count(tokenizer, cc_group_size)
        valid_predictions = []
        valid_targets = []
        strict_squared_error_sum = 0.0
        invalid_count = 0
        for completion, target in zip(completions, answer_rewards):
            token_ids = getattr(completion, "token_ids", None)
            count = token_id_to_count.get(token_ids[0]) if token_ids else None
            if count is None:
                invalid_count += 1
                strict_squared_error_sum += 1.0
                continue
            prediction = count / cc_group_size
            valid_predictions.append(prediction)
            valid_targets.append(target)
            strict_squared_error_sum += (prediction - target) ** 2

        if valid_predictions:
            errors = np.asarray(valid_predictions) - np.asarray(valid_targets)
            cc_mse = float((errors**2).mean())
        else:
            cc_mse = float("nan")
        eval_metrics.update(
            {
                "cc_mse": cc_mse,
                "cc_mse_strict": strict_squared_error_sum / len(responses),
                "cc_spearman": _spearman_correlation(valid_predictions, valid_targets),
                "cc_invalid_rate": invalid_count / len(responses),
                "cc_num_valid": len(valid_predictions),
                "cc_num_invalid": invalid_count,
            }
        )
    return eval_metrics
