"""Collection strategies for fixed-size, difficulty-filtered GRPO batches.

``RolloutBatchCollector`` owns the lifecycle shared by every strategy:
balanced row scheduling, deterministic rollout seeds, response scoring,
batch flattening, and the collect/record-trained transaction. Concrete
collectors decide when and how a candidate question is difficulty-filtered.
"""

from __future__ import annotations

import heapq
import math
import random
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, final

try:
    from .confidence_training.metrics import compute_metrics
    from .grpo_core_implementation import score_rollout_responses
except ImportError:
    from confidence_training.metrics import compute_metrics
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


class ConfidencePredictions(Protocol):
    hard_confidences: Sequence[float]
    expected_confidences: Sequence[float]
    metrics: Mapping[str, float]


class ConfidenceEstimator(Protocol):
    """Minimal interface required by the confidence-filtering collector.

    Model loading, device placement, and the choice of PyTorch or vLLM belong
    behind this interface rather than inside batch-collection logic.
    """

    def predict_confidences(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
    ) -> ConfidencePredictions: ...


@dataclass(frozen=True)
class RolloutGroup:
    training_row_index: int
    prompt: str
    ground_truth: str
    responses: list[str]
    reward_dicts: list[dict[str, float]]
    # This is always measured from the generated responses and is what GRPO
    # ultimately trains on, even when filter_score came from an estimator.
    mean_reward: float
    candidate_index: int
    # Empirical filtering sets this to mean_reward. Confidence filtering will
    # instead set it to the estimator's prompt-level predicted accuracy.
    filter_score: float
    filter_score_source: str


@dataclass(frozen=True)
class CollectedRolloutBatch:
    repeated_prompts: list[str]
    rollout_responses: list[str]
    repeated_ground_truths: list[str]
    reward_dicts: list[dict[str, float]]
    group_mean_rewards: list[float]
    group_filter_scores: list[float]
    filter_score_sources: list[str]
    training_row_indices: list[int]
    metrics: dict[str, float]


class RolloutBatchCollector(ABC):
    """Shared lifecycle for a difficulty-filtered rollout collector.

    ``collect`` and ``record_trained`` form a small transaction. Collection
    reserves the selected row indices; their persistent training counts are
    incremented only after the caller confirms that the optimizer used the
    returned batch. This prevents failed training steps from affecting future
    sampling fairness.

    Subclasses implement only ``_collect_groups``. They may use the helpers
    below to schedule rows, generate rollouts, score actual rewards, and build
    the final fixed-size batch.
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
                "difficulty bounds must satisfy "
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
        """Number of distinct rollout groups required by one training step."""

        return self.train_batch_size // self.group_size

    def _sample_training_row_indices(
        self,
        count: int,
        *,
        unavailable: set[int],
        attempt_counts: list[int],
    ) -> list[int]:
        """Select the least-trained, least-attempted available rows.

        Persistent ``training_counts`` provide fairness across optimizer steps.
        Per-collection ``attempt_counts`` spread rejected candidates across the
        dataset before retrying the same question. Random shuffling breaks ties
        reproducibly using ``scheduler_rng``.
        """

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

    def _allocate_candidate_index(self) -> int:
        """Return a globally unique candidate index and advance its counter."""

        candidate_index = self.next_candidate_index
        self.next_candidate_index += 1
        return candidate_index

    def _sampling_params_for_candidate(
        self,
        candidate_index: int,
    ) -> dict[str, Any]:
        """Give every candidate group a deterministic, non-reused seed."""

        sampling_params = dict(self.sampling_params)
        sampling_params["seed"] = self.base_seed + candidate_index
        return sampling_params

    def _submit_rollout_job(
        self,
        executor: ThreadPoolExecutor,
        *,
        training_row_index: int,
        candidate_index: int,
    ) -> Future[Sequence[Completion]]:
        """Submit generation of one question's complete rollout group."""

        row = self.train_rows[training_row_index]
        try:
            prompt = str(row["prompt"])
        except KeyError as error:
            raise ValueError("Every training row must contain 'prompt'") from error

        return executor.submit(
            self.rollout_server.generate_completions,
            prompts=[prompt],
            sampling_params=self._sampling_params_for_candidate(candidate_index),
            batch_size=None,
        )

    def _build_group(
        self,
        *,
        training_row_index: int,
        candidate_index: int,
        completions: Sequence[Completion],
        filter_score: float | None = None,
        filter_score_source: str = "empirical_reward",
    ) -> RolloutGroup:
        """Score generated responses and materialize one rollout group.

        A confidence-based collector should pass its previously predicted
        confidence as ``filter_score``. Actual response rewards are calculated
        here in either case and remain separate from that selection signal.
        """

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
            float(reward_dict["reward"]) for reward_dict in reward_dicts
        ) / self.group_size
        resolved_filter_score = mean_reward if filter_score is None else filter_score
        if not 0.0 <= resolved_filter_score <= 1.0:
            raise ValueError(
                f"filter_score must be in [0, 1], got {resolved_filter_score}"
            )
        return RolloutGroup(
            training_row_index=training_row_index,
            prompt=prompt,
            ground_truth=ground_truth,
            responses=responses,
            reward_dicts=reward_dicts,
            mean_reward=mean_reward,
            candidate_index=candidate_index,
            filter_score=resolved_filter_score,
            filter_score_source=filter_score_source,
        )

    def _flatten_groups(
        self,
        groups: Sequence[RolloutGroup],
        *,
        metrics: dict[str, float],
    ) -> CollectedRolloutBatch:
        """Convert question-level groups to the flat layout consumed by GRPO."""

        repeated_prompts = []
        rollout_responses = []
        repeated_ground_truths = []
        selected_reward_dicts = []
        for group in groups:
            repeated_prompts.extend([group.prompt] * self.group_size)
            rollout_responses.extend(group.responses)
            repeated_ground_truths.extend([group.ground_truth] * self.group_size)
            selected_reward_dicts.extend(group.reward_dicts)

        return CollectedRolloutBatch(
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            reward_dicts=selected_reward_dicts,
            group_mean_rewards=[group.mean_reward for group in groups],
            group_filter_scores=[group.filter_score for group in groups],
            filter_score_sources=[group.filter_score_source for group in groups],
            training_row_indices=[group.training_row_index for group in groups],
            metrics=metrics,
        )

    @final
    def collect(self) -> CollectedRolloutBatch:
        """Collect one batch and reserve its rows until ``record_trained``."""

        if self._pending_training_row_indices is not None:
            raise RuntimeError("Call record_trained before collecting the next batch")

        selected_groups, metrics = self._collect_groups()
        if len(selected_groups) != self.target_groups:
            raise RuntimeError(
                f"Expected {self.target_groups} selected groups, "
                f"got {len(selected_groups)}"
            )

        batch = self._flatten_groups(selected_groups, metrics=metrics)
        if len(batch.rollout_responses) != self.train_batch_size:
            raise RuntimeError(
                f"Expected {self.train_batch_size} selected rollouts, "
                f"got {len(batch.rollout_responses)}"
            )
        self._pending_training_row_indices = list(batch.training_row_indices)
        return batch

    @abstractmethod
    def _collect_groups(
        self,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Select, generate, and score exactly ``target_groups`` groups."""

    @final
    def record_trained(self, batch: CollectedRolloutBatch) -> None:
        """Record that the optimizer successfully used a collected batch."""

        if self._pending_training_row_indices is None:
            raise RuntimeError("There is no pending collected batch")
        if batch.training_row_indices != self._pending_training_row_indices:
            raise ValueError("batch does not match the pending collected batch")

        for training_row_index in batch.training_row_indices:
            self.training_counts[training_row_index] += 1
        self._pending_training_row_indices = None


class EmpiricalRewardRolloutBatchCollector(RolloutBatchCollector):
    """Filter questions using rewards measured from newly generated rollouts.

    Candidate groups are generated asynchronously. An in-range group is kept;
    a rejected group is replaced until the fixed-size batch is full or the
    candidate budget is exhausted. A bounded reservoir retains the globally
    most intermediate candidates as a deterministic fallback.
    """

    def _collect_groups(
        self,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        accepted_groups: list[RolloutGroup] = []
        intermediate_reservoir: list[tuple[float, int, RolloutGroup]] = []
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
            """Keep one generation job in flight per still-needed group."""

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
                candidate_index = self._allocate_candidate_index()
                future = self._submit_rollout_job(
                    executor,
                    training_row_index=training_row_index,
                    candidate_index=candidate_index,
                )
                in_flight[future] = (training_row_index, candidate_index)
            max_inflight_jobs = max(max_inflight_jobs, len(in_flight))

        with ThreadPoolExecutor(max_workers=self.target_groups) as executor:
            fill_available_slots(executor)
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                # Concurrent jobs can finish nondeterministically. Processing
                # each completed wave in dispatch order preserves stable output.
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

                    # heapq keeps the worst retained item at index zero. The
                    # negative keys therefore keep groups closest to 0.5, with
                    # earlier candidate indices winning exact ties.
                    distance = abs(group.filter_score - 0.5)
                    reservoir_item = (-distance, -group.candidate_index, group)
                    if len(intermediate_reservoir) < self.target_groups:
                        heapq.heappush(intermediate_reservoir, reservoir_item)
                    elif reservoir_item > intermediate_reservoir[0]:
                        heapq.heapreplace(intermediate_reservoir, reservoir_item)

                    if self.lower_bound <= group.filter_score <= self.upper_bound:
                        in_range_groups += 1
                        accepted_groups.append(group)

                fill_available_slots(executor)

        used_intermediate_fallback = len(accepted_groups) < self.target_groups
        if used_intermediate_fallback:
            if len(intermediate_reservoir) < self.target_groups:
                raise RuntimeError(
                    "Candidate budget ended before enough groups were generated"
                )
            selected_groups = sorted(
                (item[2] for item in intermediate_reservoir),
                key=lambda group: (
                    abs(group.filter_score - 0.5),
                    group.candidate_index,
                ),
            )
        else:
            selected_groups = sorted(
                accepted_groups,
                key=lambda group: group.candidate_index,
            )

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
            "candidate_mean_reward": candidate_reward_sum / candidate_groups,
            "selected_mean_reward": selected_mean_reward,
            "max_inflight_jobs": float(max_inflight_jobs),
            "used_intermediate_fallback": float(used_intermediate_fallback),
            "selected_out_of_range_questions": float(
                sum(
                    not self.lower_bound
                    <= group.filter_score
                    <= self.upper_bound
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
        return selected_groups, metrics


class ConfidenceEstimatorRolloutBatchCollector(RolloutBatchCollector):
    """Filter candidate questions before rollout using predicted confidence."""

    def __init__(
        self,
        *,
        rollout_server: RolloutServer,
        train_rows: Sequence[Mapping[str, Any]],
        reward_fn: Callable[[str, str], dict[str, float]],
        confidence_estimator: ConfidenceEstimator,
        sampling_params: Mapping[str, Any],
        train_batch_size: int,
        group_size: int,
        lower_bound: float,
        upper_bound: float,
        max_candidate_groups: int,
        confidence_batch_size: int,
        scheduler_seed: int | None = None,
    ) -> None:
        super().__init__(
            rollout_server=rollout_server,
            train_rows=train_rows,
            reward_fn=reward_fn,
            sampling_params=sampling_params,
            train_batch_size=train_batch_size,
            group_size=group_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_candidate_groups=max_candidate_groups,
            scheduler_seed=scheduler_seed,
        )
        if confidence_batch_size <= 0:
            raise ValueError("confidence_batch_size must be positive")
        self.confidence_estimator = confidence_estimator
        self.confidence_batch_size = confidence_batch_size
        self.next_training_row_index = 0
        self.collection_step = 0

    def _next_candidate_row_index(
        self,
        *,
        cursor: int,
        unavailable: set[int],
        attempt_counts: Sequence[int],
    ) -> tuple[int, int]:
        available = [
            index
            for index in range(len(self.train_rows))
            if index not in unavailable
        ]
        if not available:
            raise RuntimeError("No unexamined training rows remain")
        minimum_key = min(
            (self.training_counts[index], attempt_counts[index])
            for index in available
        )
        for offset in range(len(self.train_rows)):
            index = (cursor + offset) % len(self.train_rows)
            if index in unavailable:
                continue
            key = (self.training_counts[index], attempt_counts[index])
            if key == minimum_key:
                return index, (index + 1) % len(self.train_rows)
        raise RuntimeError("Could not choose the next confidence candidate")

    def _validate_confidences(
        self,
        values: Sequence[float],
        *,
        name: str,
    ) -> list[float]:
        if len(values) != len(self.train_rows):
            raise ValueError(
                f"Expected {len(self.train_rows)} {name} values, got {len(values)}"
            )
        converted = [float(value) for value in values]
        invalid = [
            value
            for value in converted
            if not math.isfinite(value) or not 0.0 <= value <= 1.0
        ]
        if invalid:
            raise ValueError(f"{name} values must be finite and in [0, 1]")
        return converted

    def _collect_groups(
        self,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        import time
        start = time.perf_counter()
        # XXX
        predictions = self.confidence_estimator.predict_confidences(
            [str(row["prompt"]) for row in self.train_rows],
            batch_size=self.confidence_batch_size,
        )
        hard_confidences = self._validate_confidences(
            predictions.hard_confidences,
            name="hard confidence",
        )
        expected_confidences = self._validate_confidences(
            predictions.expected_confidences,
            name="expected confidence",
        )
        # XXX
        elapsed = time.perf_counter() - start
        print(f"Confidence estimation done {elapsed:.2f} seconds ({len(hard_confidences)} hard confidences, mean = {sum(hard_confidences) / len(hard_confidences)}; {len(expected_confidences)} expected confidences, mean = {sum(expected_confidences) / len(expected_confidences)})")  # XXX

        accepted_candidates: list[tuple[int, int, float, float]] = []
        intermediate_reservoir: list[
            tuple[float, int, tuple[int, int, float, float]]
        ] = []
        attempt_counts = [0] * len(self.train_rows)
        considered_rows: set[int] = set()
        candidate_cursor = self.next_training_row_index
        in_range_candidates = 0
        candidate_score_sum = 0.0
        candidate_budget = min(self.max_candidate_groups, len(self.train_rows))

        # XXX
        print(f"Candidate budget = {candidate_budget}")
        start = time.perf_counter()
        while (
            len(accepted_candidates) < self.target_groups
            and len(considered_rows) < candidate_budget
        ):
            training_row_index, candidate_cursor = self._next_candidate_row_index(
                cursor=candidate_cursor,
                unavailable=considered_rows,
                attempt_counts=attempt_counts,
            )
            considered_rows.add(training_row_index)
            attempt_counts[training_row_index] += 1
            candidate_index = self._allocate_candidate_index()
            hard_confidence = hard_confidences[training_row_index]
            expected_confidence = expected_confidences[training_row_index]
            candidate = (
                training_row_index,
                candidate_index,
                hard_confidence,
                expected_confidence,
            )
            candidate_score_sum += expected_confidence

            reservoir_item = (
                -abs(expected_confidence - 0.5),
                -candidate_index,
                candidate,
            )
            if len(intermediate_reservoir) < self.target_groups:
                heapq.heappush(intermediate_reservoir, reservoir_item)
            elif reservoir_item > intermediate_reservoir[0]:
                heapq.heapreplace(intermediate_reservoir, reservoir_item)

            if self.lower_bound <= expected_confidence <= self.upper_bound:
                in_range_candidates += 1
                accepted_candidates.append(candidate)

        used_intermediate_fallback = (
            len(accepted_candidates) < self.target_groups
        )
        if used_intermediate_fallback:
            if len(intermediate_reservoir) < self.target_groups:
                raise RuntimeError(
                    "Candidate budget ended before enough confidence candidates "
                    "were considered"
                )
            selected_candidates = sorted(
                (item[2] for item in intermediate_reservoir),
                key=lambda candidate: (
                    abs(candidate[3] - 0.5),
                    candidate[1],
                ),
            )
        else:
            selected_candidates = sorted(
                accepted_candidates,
                key=lambda candidate: candidate[1],
            )
        # XXX
        elapsed = time.perf_counter() - start
        print(f"{elapsed:.2f} seconds elapsed for candidate selection")
        selected_prompts = [
            str(self.train_rows[candidate[0]]["prompt"])
            for candidate in selected_candidates
        ]
        rollout_sampling_params = dict(self.sampling_params)
        rollout_sampling_params["seed"] = (
            self.base_seed + self.collection_step * self.target_groups
        )
        # XXX
        start = time.perf_counter()
        completions = self.rollout_server.generate_completions(
            prompts=selected_prompts,
            sampling_params=rollout_sampling_params,
            batch_size=None,
        )
        elapsed = time.perf_counter() - start
        print(f"{elapsed:.2f} seconds for rollout generation")
        expected_completion_count = self.target_groups * self.group_size
        if len(completions) != expected_completion_count:
            raise RuntimeError(
                f"Expected {expected_completion_count} completions, "
                f"got {len(completions)}"
            )

        selected_groups: list[RolloutGroup] = []
        selected_hard_confidences: list[float] = []
        selected_expected_confidences: list[float] = []
        for group_index, candidate in enumerate(selected_candidates):
            training_row_index, candidate_index, hard_confidence, expected_confidence = candidate
            start = group_index * self.group_size
            group = self._build_group(
                training_row_index=training_row_index,
                candidate_index=candidate_index,
                completions=completions[start : start + self.group_size],
                filter_score=expected_confidence,
                filter_score_source="confidence_estimator_expected",
            )
            selected_groups.append(group)
            selected_hard_confidences.append(hard_confidence)
            selected_expected_confidences.append(expected_confidence)

        empirical_targets = [group.mean_reward for group in selected_groups]
        hard_metrics = compute_metrics(
            selected_hard_confidences,
            empirical_targets,
        )
        expected_metrics = compute_metrics(
            selected_expected_confidences,
            empirical_targets,
        )
        candidate_questions = len(considered_rows)
        metrics = {
            "confidence_scored_questions": float(len(self.train_rows)),
            "candidate_questions": float(candidate_questions),
            "candidate_rollouts": float(expected_completion_count),
            "unique_candidate_questions": float(candidate_questions),
            "in_range_questions": float(in_range_candidates),
            "selected_questions": float(self.target_groups),
            "acceptance_rate": in_range_candidates / candidate_questions,
            "candidate_mean_filter_score": candidate_score_sum / candidate_questions,
            "selected_mean_filter_score": sum(selected_expected_confidences)
            / self.target_groups,
            "selected_mean_hard_confidence": sum(selected_hard_confidences)
            / self.target_groups,
            "selected_mean_reward": sum(empirical_targets) / self.target_groups,
            "used_intermediate_fallback": float(used_intermediate_fallback),
            "selected_out_of_range_questions": float(
                sum(
                    not self.lower_bound
                    <= candidate[3]
                    <= self.upper_bound
                    for candidate in selected_candidates
                )
            ),
            "max_attempts_per_question": float(max(attempt_counts)),
            "training_count_min_before": float(min(self.training_counts)),
            "training_count_max_before": float(max(self.training_counts)),
            "training_count_mean_before": sum(self.training_counts)
            / len(self.training_counts),
            "calibration/mse": float(hard_metrics["mse"]),
            "calibration/spearman_r": float(hard_metrics["spearman_r"]),
            "calibration/expected_mse": float(expected_metrics["mse"]),
            "calibration/expected_spearman_r": float(
                expected_metrics["spearman_r"]
            ),
        }
        metrics.update(
            {
                f"confidence_{name}": float(value)
                for name, value in predictions.metrics.items()
            }
        )
        self.next_training_row_index = candidate_cursor
        self.collection_step += 1
        return selected_groups, metrics
