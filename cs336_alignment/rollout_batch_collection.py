"""Asynchronous collection of fixed-size GRPO batches with reward filtering."""

from __future__ import annotations

import heapq
import random
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from .grpo_core_implementation import score_rollout_responses
except ImportError:
    from grpo_core_implementation import score_rollout_responses


class Completion(Protocol):
    text: str


class RolloutServer(Protocol):
    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any],
        batch_size: int | None = None,
    ) -> Sequence[Completion]: ...


@dataclass(frozen=True)
class RolloutGroup:
    training_row_index: int
    prompt: str
    ground_truth: str
    responses: list[str]
    reward_dicts: list[dict[str, float]]
    mean_reward: float
    candidate_index: int


@dataclass(frozen=True)
class CollectedRolloutBatch:
    repeated_prompts: list[str]
    rollout_responses: list[str]
    repeated_ground_truths: list[str]
    reward_dicts: list[dict[str, float]]
    group_mean_rewards: list[float]
    training_row_indices: list[int]
    metrics: dict[str, float]


class EmpiricalRewardRolloutBatchCollector:
    """Collect on-policy groups using asynchronous, balanced sampling.

    Persistent training counts increase only after ``record_trained`` confirms
    that a collected batch was used for an optimizer step. Within a collection
    step, temporary attempt counts spread rejected attempts across equally
    under-trained examples.
    """

    def __init__(
        self,
        *,
        rollout_server: RolloutServer,
        train_rows: Sequence[Mapping[str, Any]],
        reward_fn: Callable[[str, str], dict[str, float]],
        sampling_params: Mapping[str, Any],
        train_batch_size: int,
        group_size: int,
        lower_bound: float,
        upper_bound: float,
        max_candidate_groups: int,
        scheduler_seed: int | None = None,
    ) -> None:
        if train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if train_batch_size % group_size != 0:
            raise ValueError("train_batch_size must be divisible by group_size")
        if not 0.0 <= lower_bound <= upper_bound <= 1.0:
            raise ValueError(
                "reward bounds must satisfy "
                "0 <= lower_bound <= upper_bound <= 1"
            )
        if sampling_params.get("n") != group_size:
            raise ValueError("sampling_params['n'] must equal group_size")

        target_groups = train_batch_size // group_size
        if len(train_rows) < target_groups:
            raise ValueError(
                f"Need at least {target_groups} training rows, got {len(train_rows)}"
            )
        if max_candidate_groups < target_groups:
            raise ValueError(
                "max_candidate_groups must be at least the target group count "
                f"({target_groups})"
            )

        base_seed = sampling_params.get("seed", 0)
        if not isinstance(base_seed, int):
            raise ValueError("sampling_params['seed'] must be an integer")

        self.rollout_server = rollout_server
        self.train_rows = train_rows
        self.reward_fn = reward_fn
        self.sampling_params = dict(sampling_params)
        self.train_batch_size = train_batch_size
        self.group_size = group_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_candidate_groups = max_candidate_groups
        self.base_seed = base_seed
        self.scheduler_rng = random.Random(
            base_seed if scheduler_seed is None else scheduler_seed
        )
        self.training_counts = [0] * len(train_rows)
        self.next_candidate_index = 0
        self._pending_training_row_indices: list[int] | None = None

    @property
    def target_groups(self) -> int:
        return self.train_batch_size // self.group_size

    def _sample_training_row_indices(
        self,
        count: int,
        *,
        unavailable: set[int],
        attempt_counts: list[int],
    ) -> list[int]:
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, training_count in enumerate(self.training_counts):
            if index in unavailable:
                continue
            key = (training_count, attempt_counts[index])
            buckets.setdefault(key, []).append(index)

        selected = []
        for key in sorted(buckets):
            candidates = buckets[key]
            self.scheduler_rng.shuffle(candidates)
            needed = count - len(selected)
            selected.extend(candidates[:needed])
            if len(selected) == count:
                return selected

        raise RuntimeError(
            f"Could only schedule {len(selected)} of {count} requested jobs"
        )

    def _submit_job(
        self,
        executor: ThreadPoolExecutor,
        *,
        training_row_index: int,
        candidate_index: int,
    ) -> Future[Sequence[Completion]]:
        row = self.train_rows[training_row_index]
        try:
            prompt = str(row["prompt"])
        except KeyError as error:
            raise ValueError("Every training row must contain 'prompt'") from error

        sampling_params = dict(self.sampling_params)
        sampling_params["seed"] = self.base_seed + candidate_index
        return executor.submit(
            self.rollout_server.generate_completions,
            prompts=[prompt],
            sampling_params=sampling_params,
            batch_size=None,
        )

    def _build_group(
        self,
        *,
        training_row_index: int,
        candidate_index: int,
        completions: Sequence[Completion],
    ) -> RolloutGroup:
        row = self.train_rows[training_row_index]
        try:
            prompt = str(row["prompt"])
            ground_truth = str(row["answer"])
        except KeyError as error:
            raise ValueError(
                f"Every training row must contain {error.args[0]!r}"
            ) from error
        if len(completions) != self.group_size:
            raise RuntimeError(
                f"Expected {self.group_size} completions, got {len(completions)}"
            )

        responses = [completion.text for completion in completions]
        reward_dicts = score_rollout_responses(
            self.reward_fn,
            responses,
            [ground_truth] * self.group_size,
        )
        mean_reward = sum(
            float(reward_dict["reward"])
            for reward_dict in reward_dicts
        ) / self.group_size
        return RolloutGroup(
            training_row_index=training_row_index,
            prompt=prompt,
            ground_truth=ground_truth,
            responses=responses,
            reward_dicts=reward_dicts,
            mean_reward=mean_reward,
            candidate_index=candidate_index,
        )

    def _flatten_groups(
        self,
        groups: list[RolloutGroup],
        *,
        metrics: dict[str, float],
    ) -> CollectedRolloutBatch:
        repeated_prompts = []
        rollout_responses = []
        repeated_ground_truths = []
        selected_reward_dicts = []
        for group in groups:
            repeated_prompts.extend([group.prompt] * self.group_size)
            rollout_responses.extend(group.responses)
            repeated_ground_truths.extend(
                [group.ground_truth] * self.group_size
            )
            selected_reward_dicts.extend(group.reward_dicts)

        return CollectedRolloutBatch(
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            reward_dicts=selected_reward_dicts,
            group_mean_rewards=[group.mean_reward for group in groups],
            training_row_indices=[
                group.training_row_index for group in groups
            ],
            metrics=metrics,
        )

    def collect(self) -> CollectedRolloutBatch:
        if self._pending_training_row_indices is not None:
            raise RuntimeError(
                "Call record_trained before collecting the next batch"
            )

        accepted_groups: list[RolloutGroup] = []
        intermediate_reservoir: list[
            tuple[float, int, RolloutGroup]
        ] = []
        attempt_counts = [0] * len(self.train_rows)
        candidate_groups = 0
        in_range_groups = 0
        candidate_reward_sum = 0.0
        max_inflight_jobs = 0
        unique_training_rows: set[int] = set()

        in_flight: dict[
            Future[Sequence[Completion]],
            tuple[int, int],
        ] = {}

        def fill_available_slots(executor: ThreadPoolExecutor) -> None:
            nonlocal max_inflight_jobs
            desired_jobs = self.target_groups - len(accepted_groups)
            available_slots = desired_jobs - len(in_flight)
            remaining_budget = self.max_candidate_groups - (
                candidate_groups + len(in_flight)
            )
            jobs_to_submit = min(available_slots, remaining_budget)
            if jobs_to_submit <= 0:
                return

            selected_indices = self._sample_training_row_indices(
                jobs_to_submit,
                unavailable={
                    training_row_index
                    for training_row_index, _ in in_flight.values()
                },
                attempt_counts=attempt_counts,
            )
            for training_row_index in selected_indices:
                attempt_counts[training_row_index] += 1
                candidate_index = self.next_candidate_index
                self.next_candidate_index += 1
                future = self._submit_job(
                    executor,
                    training_row_index=training_row_index,
                    candidate_index=candidate_index,
                )
                in_flight[future] = (
                    training_row_index,
                    candidate_index,
                )
            max_inflight_jobs = max(max_inflight_jobs, len(in_flight))

        with ThreadPoolExecutor(max_workers=self.target_groups) as executor:
            fill_available_slots(executor)
            while in_flight:
                completed, _ = wait(
                    in_flight,
                    return_when=FIRST_COMPLETED,
                )
                completed_in_dispatch_order = sorted(
                    completed,
                    key=lambda future: in_flight[future][1],
                )
                for future in completed_in_dispatch_order:
                    training_row_index, candidate_index = in_flight.pop(future)
                    group = self._build_group(
                        training_row_index=training_row_index,
                        candidate_index=candidate_index,
                        completions=future.result(),
                    )
                    candidate_groups += 1
                    unique_training_rows.add(training_row_index)
                    candidate_reward_sum += group.mean_reward

                    distance = abs(group.mean_reward - 0.5)
                    reservoir_item = (
                        -distance,
                        -group.candidate_index,
                        group,
                    )
                    if len(intermediate_reservoir) < self.target_groups:
                        heapq.heappush(
                            intermediate_reservoir,
                            reservoir_item,
                        )
                    elif reservoir_item > intermediate_reservoir[0]:
                        heapq.heapreplace(
                            intermediate_reservoir,
                            reservoir_item,
                        )

                    if (
                        self.lower_bound
                        <= group.mean_reward
                        <= self.upper_bound
                    ):
                        in_range_groups += 1
                        accepted_groups.append(group)

                fill_available_slots(executor)

        used_intermediate_fallback = (
            len(accepted_groups) < self.target_groups
        )
        if used_intermediate_fallback:
            if len(intermediate_reservoir) < self.target_groups:
                raise RuntimeError(
                    "Candidate budget ended before enough groups were generated"
                )
            selected_groups = sorted(
                (item[2] for item in intermediate_reservoir),
                key=lambda group: (
                    abs(group.mean_reward - 0.5),
                    group.candidate_index,
                ),
            )
        else:
            selected_groups = sorted(
                accepted_groups,
                key=lambda group: group.candidate_index,
            )

        if len(selected_groups) != self.target_groups:
            raise RuntimeError(
                f"Expected {self.target_groups} selected groups, "
                f"got {len(selected_groups)}"
            )

        selected_row_indices = [
            group.training_row_index for group in selected_groups
        ]
        selected_mean_reward = sum(
            group.mean_reward for group in selected_groups
        ) / self.target_groups
        metrics = {
            "candidate_questions": float(candidate_groups),
            "candidate_rollouts": float(candidate_groups * self.group_size),
            "unique_candidate_questions": float(len(unique_training_rows)),
            "in_range_questions": float(in_range_groups),
            "selected_questions": float(self.target_groups),
            "acceptance_rate": in_range_groups / candidate_groups,
            "candidate_mean_reward": (
                candidate_reward_sum / candidate_groups
            ),
            "selected_mean_reward": selected_mean_reward,
            "max_inflight_jobs": float(max_inflight_jobs),
            "used_intermediate_fallback": float(
                used_intermediate_fallback
            ),
            "selected_out_of_range_questions": float(
                sum(
                    not (
                        self.lower_bound
                        <= group.mean_reward
                        <= self.upper_bound
                    )
                    for group in selected_groups
                )
            ),
            "max_attempts_per_question": float(max(attempt_counts)),
            "training_count_min_before": float(min(self.training_counts)),
            "training_count_max_before": float(max(self.training_counts)),
            "training_count_mean_before": (
                sum(self.training_counts) / len(self.training_counts)
            ),
        }
        self._pending_training_row_indices = selected_row_indices
        return self._flatten_groups(selected_groups, metrics=metrics)

    def record_trained(self, batch: CollectedRolloutBatch) -> None:
        """Record that the optimizer successfully used a collected batch."""
        if self._pending_training_row_indices is None:
            raise RuntimeError("There is no pending collected batch")
        if batch.training_row_indices != self._pending_training_row_indices:
            raise ValueError("batch does not match the pending collected batch")

        for training_row_index in batch.training_row_indices:
            self.training_counts[training_row_index] += 1
        self._pending_training_row_indices = None
