from typing import Callable, Sequence, Mapping, Any
from vllm_utils import VLLMServer

def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    try:
        return str(completion.text)
    except AttributeError as error:
        raise TypeError("Completions must be strings or objects with a 'text' attribute.") from error

def evaluate(
    vllm_server: VLLMServer,
    eval_dataset: Sequence[Mapping[str, Any]],
    sampling_params: dict,
    batch_size: int,
    reward_fn: Callable,
):
    print(f"Evaluating {len(eval_dataset)} instances ...")
    rows = list(eval_dataset)
    if not rows:
        raise ValueError(
            "valid_dataset is empty; check the configured validation dataset path "
            "and train/validation split sizes."
        )
    if sampling_params.get("n") != 1:
        raise ValueError("Evaluation requires sampling_params['n'] == 1.")
    prompts = [row["prompt"] for row in rows]
    ground_truths = [row["answer"] for row in rows]
    completions = vllm_server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    responses = [_completion_text(completion) for completion in completions]
    if not (len(prompts) == len(ground_truths) == len(responses)):
        raise ValueError(f"The length of prompts ({len(prompts)}) or ground_truths ({len(ground_truths)}) should be identical to the length of resopnses ({len(responses)})")
    eval_metrics = None
    for response, ground_truth in zip(responses, ground_truths):
        metrics = reward_fn(response=response, ground_truth=ground_truth)
        if eval_metrics is None:
            eval_metrics = metrics
        else:
            for k, v in metrics.items():
                eval_metrics[k] += v
    # Average the metrics
    for k in eval_metrics.keys():
        eval_metrics[k] /= len(responses)
    return eval_metrics
