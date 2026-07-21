from typing import Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    prompt_tokens = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    sequences = list()
    response_mask = list()
    # Construct the correct prompt_and_output concatenated sequence first
    for prompt, output in zip(prompt_tokens, output_tokens):
        sequences.append(prompt + output)  # We would shift 1 position at the end when returning
        response_mask.append([0] * len(prompt) + [1] * len(output))  # We would shift 1 position at the end when returning
    # Padding later
    max_seq_len = max(len(sequence) for sequence in sequences)
    for i in range(len(sequences)):
        padding_len = max_seq_len - len(sequences[i])
        if padding_len > 0:
            sequences[i] += [tokenizer.pad_token_id] * padding_len
            response_mask[i] += [0] * padding_len
    return {
        "input_ids": torch.tensor(sequences, dtype=torch.long)[:, :-1],
        "labels": torch.tensor(sequences, dtype=torch.long)[:, 1:],
        "response_mask": torch.tensor(response_mask, dtype=torch.long)[:, 1:]
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    outputs = dict()
    logits = model(input_ids).logits
    # ===== Implementation 1 =====
    # NOTE: logits must be normalized with before they become log-probs.
    all_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    label_log_probs = all_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    outputs["log_probs"] = label_log_probs
    if return_token_entropy:
        outputs["token_entropy"] = -(all_log_probs.exp() * all_log_probs).sum(-1)
    # ===== End of Implementation 1 =====
    # # ===== Implementation 2: originally done for better memory efficiency, but it turns out slower empirically (might be due to optimized kernel used in Implmenetation 1) =====
    #     # NOTE: logits must be normalized with before they become log-probs.
    # if return_token_entropy:
    #     all_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    #     label_log_probs = all_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    #     outputs["log_probs"] = label_log_probs
    #     outputs["token_entropy"] = -(all_log_probs.exp() * all_log_probs).sum(-1)
    # else:  # NOTE: we avoid materializing all log-probs, but still compute the full-vocab normalizer
    #     label_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    #     normalizing_logits = logits.logsumexp(dim=-1) # NOTE: it is calculating max_logits + log(sum(exp(logits - max_logits))) under the hood
    #     outputs["log_probs"] = label_logits - normalizing_logits
    # # ===== End of Implementation 2 =====
    return outputs


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    assert len(rollout_responses) == len(repeated_ground_truths)
    reward_dicts = [
        reward_fn(res, gt)
        for res, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor([reward_dict["reward"] for reward_dict in reward_dicts])
    format_rewards = torch.tensor([reward_dict["format_reward"] for reward_dict in reward_dicts])
    metadata = {
        "mean_total_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item()
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    n_prompts = len(raw_rewards) / group_size
    assert n_prompts.is_integer()
    advantages = torch.empty(len(raw_rewards))
    for i in range(int(n_prompts)):
        group_rewards = raw_rewards[i * group_size : (i + 1) * group_size]
        match baseline:
            case "mean":
                b = torch.mean(group_rewards)
            case "none":
                b = 0
            case _:
                raise ValueError(f"baseline value {baseline} not supported")
        match advantage_normalizer:
            case "std":
                normalizer = torch.std(group_rewards) + advantage_eps
            case "none":
                normalizer = 1.0
            case "mean":
                normalizer = torch.mean(group_rewards) + advantage_eps
        advantages[i * group_size : (i + 1) * group_size] = (group_rewards - b) / normalizer
    metadata = dict()  # TODO: implement required metadata in the future
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    assert raw_rewards_or_advantages.shape[0] == policy_log_probs.shape[0]
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)
    # NOTE: since PyTorch optimizers do gradient descent, don't forget to multiply by -1!
    per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    metadata = dict()  # TODO: put in useful information later
    match importance_reweighting_method:
        case "none":
            pass
        case "noclip":  # TODO: apply importance reweighting without clipping
            raise NotImplementedError
        case "grpo":  # TODO: do PPO/GRPO-style token-level reweighting and clipping
            raise NotImplementedError
        case "gspo":  # TODO: do GSPO-style sequence-level reweighting and clipping
            raise NotImplementedError
    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    assert per_token_policy_gradient_loss.shape == mask.shape
    masked_loss = mask * per_token_policy_gradient_loss  # shape = (batch_size, sequence_length)
    match loss_normalization:
        case "sequence":
            normalized_loss_of_each_sequence = masked_loss.sum(dim=-1) / mask.sum(dim=-1)
            avg_loss = normalized_loss_of_each_sequence.sum() / masked_loss.shape[0]  # NOTE: remember to normalize by batch_size (i.e., B * G)
        case "constant":
            raise NotImplementedError
    return avg_loss


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    # rewards should be computed outside of the gradient accumulation loop to make calculations of group-level stats (e.g., mean and std) simple
    raw_rewards, raw_rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantages_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    assert len(repeated_prompts) == len(rollout_responses) == len(repeated_ground_truths)
    batch_size = len(repeated_prompts)
    microbatch_size = batch_size // gradient_accumulation_steps  # NOTE: it's okay if batch_size / gradient_accumulation_steps is not an integer (see the note below)
    # Gradient accumulation across microbatches
    loss = torch.tensor(0.0)
    avg_entropy = torch.tensor(0.0)  # TODO: check why this information might be useful and whether my implementation is reasonable
    for i in range(gradient_accumulation_steps):
        # Get the data related to this microbatch
        prompts = repeated_prompts[i*microbatch_size:(i+1)*microbatch_size]
        responses = rollout_responses[i*microbatch_size:(i+1)*microbatch_size]
        microbatch_advantages = advantages[i*microbatch_size:(i+1)*microbatch_size]
        microbatch_old_log_probs = old_log_probs[i*microbatch_size:(i+1)*microbatch_size] if old_log_probs else None
        # Tokenization (on CPU)
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)  # a dict with keys "input_ids", "labels", and "response_mask"
        # Get each prefix-conditioned response token's logprob (and optionally entropy), which is necessary for computing policy gradients
        model_device = next(model.parameters()).device  # NOTE: this assume that all parameters of the model are on the same GPU
        response_token_logprobs = get_response_log_probs(  # two keys: ["log_probs", "token_entropy"], shape of the logprobs: (microbatch_size, max_sequence_length_in_this_microbatch)
            model,
            input_ids=tokenized["input_ids"].to(model_device),
            labels=tokenized["labels"].to(model_device),  # Remember to move the tokenized data to the same device as the model
            return_token_entropy=True
        )
        avg_entropy += response_token_logprobs["token_entropy"].mean().detach().cpu() * (len(prompts) / batch_size)
        # Compute policy-gradient loss
        per_token_loss, per_token_loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=microbatch_advantages.to(model_device),  # NOTE: remember to move to the same device!
            policy_log_probs=response_token_logprobs["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=microbatch_old_log_probs.to(model_device) if microbatch_old_log_probs else None,
            cliprange=cliprange,
            response_mask=tokenized["response_mask"].to(model_device)
        )
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=tokenized["response_mask"].to(model_device),
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        ) * (per_token_loss.shape[0] / batch_size)  # NOTE: Important!!! Remember to rescale the loss to make the accumulated loss equivalent to the full-batch loss
        microbatch_loss.backward()
        loss += microbatch_loss.detach().cpu()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    metadata={
        "gradient_norm": total_norm,
        "token_entropy": avg_entropy,
        "mean_total_reward": raw_rewards_metadata["mean_total_reward"],
        "mean_format_reward": raw_rewards_metadata["mean_format_reward"],
    }
    return loss, metadata
