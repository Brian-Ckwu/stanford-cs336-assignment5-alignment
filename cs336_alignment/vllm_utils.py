"""
Small vLLM helpers for server lifecycle, completion requests, and weight sync.
"""

import atexit
import json
import logging
import math
import os
import signal
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


@dataclass(frozen=True)
class VLLMConfidencePredictions:
    hard_confidences: list[float]
    expected_confidences: list[float]
    metrics: dict[str, float]


@dataclass
class VLLMServer:
    model_id: str
    host: str = "127.0.0.1"
    port: int | None = None
    gpu: int = 1
    seed: int = 0
    load_format: str = "auto"
    logging_level: str = "ERROR"
    gpu_memory_utilization: float = 0.9
    weight_transfer_backend: Literal["nccl", "ipc"] = "nccl"
    enable_lora: bool = False
    max_lora_rank: int = 16
    max_loras: int = 1
    max_model_len: int = 1024
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None
    launch_server: bool = True
    startup_timeout: int = 600
    shutdown_timeout: int = 30

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if (
            self.max_num_batched_tokens is not None
            and self.max_num_batched_tokens <= 0
        ):
            raise ValueError("max_num_batched_tokens must be positive")
        if self.port is None:
            self.port = find_available_port(self.host)
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        self.weight_sync_group = None
        self.served_model_id = self.model_id

    def start(self) -> None:
        if self.launch_server:
            self.process = start_server(
                model_id=self.model_id,
                host=self.host,
                port=self.port,
                gpu=self.gpu,
                seed=self.seed,
                load_format=self.load_format,
                logging_level=self.logging_level,
                gpu_memory_utilization=self.gpu_memory_utilization,
                weight_transfer_backend=self.weight_transfer_backend,
                enable_lora=self.enable_lora,
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
            )
            atexit.register(self.stop)
        wait_for_server(self.base_url, self.process, self.startup_timeout)

    def stop(self) -> None:
        stop_server(self.process, timeout=self.shutdown_timeout)

    def init_weight_sync(self, policy_device: str):
        if self.weight_transfer_backend == "ipc":
            self.weight_sync_group = None
            return None
        self.weight_sync_group = init_weight_sync(self.base_url, policy_device)
        return self.weight_sync_group

    def sync_policy_weights(self, policy: torch.nn.Module) -> None:
        if self.weight_transfer_backend == "ipc":
            sync_policy_weights_ipc(policy, self.base_url)
            return
        if self.weight_sync_group is None:
            raise RuntimeError("Call init_weight_sync before sync_policy_weights.")
        sync_policy_weights(policy, self.base_url, self.weight_sync_group)

    def load_lora_adapter(
        self,
        name: str,
        path: str,
        *,
        load_inplace: bool = False,
        set_default: bool = True,
    ) -> None:
        if not self.enable_lora:
            raise RuntimeError("Start the vLLM server with enable_lora=True.")
        _http_json(
            "POST",
            f"{self.base_url}/v1/load_lora_adapter",
            {
                "lora_name": name,
                "lora_path": os.path.abspath(path),
                "load_inplace": load_inplace,
            },
            timeout=300,
        )
        # Prefix-cache entries produced by older adapter weights are stale.
        _http_json("POST", f"{self.base_url}/reset_prefix_cache", timeout=60)
        if set_default:
            self.served_model_id = name

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
        model_id: str | None = None,
    ) -> list[VLLMCompletion]:
        return generate_completions(
            vllm_base_url=self.base_url,
            model_id=self.served_model_id if model_id is None else model_id,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )

    def generate_confidence_predictions(
        self,
        prompts: Sequence[str],
        *,
        model_id: str,
        candidate_token_ids: Sequence[int],
        group_size: int,
        batch_size: int,
    ) -> VLLMConfidencePredictions:
        return generate_confidence_predictions(
            vllm_base_url=self.base_url,
            model_id=model_id,
            prompts=prompts,
            candidate_token_ids=candidate_token_ids,
            group_size=group_size,
            batch_size=batch_size,
            seed=self.seed,
        )


@dataclass
class VLLMConfidenceEstimator:
    server: VLLMServer
    adapter_name: str
    candidate_token_ids: tuple[int, ...]
    group_size: int
    cache_predictions: bool = False
    _cached_prompts: tuple[str, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_predictions: VLLMConfidencePredictions | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def clear_cache(self) -> None:
        self._cached_prompts = None
        self._cached_predictions = None

    def _result_with_cache_metrics(
        self,
        predictions: VLLMConfidencePredictions,
        *,
        cache_hit: bool,
    ) -> VLLMConfidencePredictions:
        metrics = dict(predictions.metrics)
        metrics["cache_enabled"] = float(self.cache_predictions)
        metrics["cache_hit"] = float(cache_hit)
        if cache_hit:
            metrics["inference_seconds"] = 0.0
        return VLLMConfidencePredictions(
            hard_confidences=list(predictions.hard_confidences),
            expected_confidences=list(predictions.expected_confidences),
            metrics=metrics,
        )

    def predict_confidences(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
    ) -> VLLMConfidencePredictions:
        prompt_key = tuple(prompts)
        if (
            self.cache_predictions
            and prompt_key == self._cached_prompts
            and self._cached_predictions is not None
        ):
            return self._result_with_cache_metrics(
                self._cached_predictions,
                cache_hit=True,
            )

        predictions = self.server.generate_confidence_predictions(
            prompt_key,
            model_id=self.adapter_name,
            candidate_token_ids=self.candidate_token_ids,
            group_size=self.group_size,
            batch_size=batch_size,
        )
        if self.cache_predictions:
            self._cached_prompts = prompt_key
            self._cached_predictions = predictions
        return self._result_with_cache_metrics(
            predictions,
            cache_hit=False,
        )


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        # Some successful vLLM endpoints, including LoRA loading, return
        # plain-text confirmation rather than JSON.
        return {"message": body.decode("utf-8", errors="replace")}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def find_available_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def kill_existing_vllm_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        result = subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        if result.returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass


def start_server(
    model_id: str,
    host: str,
    port: int,
    gpu: int,
    seed: int,
    load_format: str,
    logging_level: str,
    gpu_memory_utilization: float = 0.9,
    weight_transfer_backend: Literal["nccl", "ipc"] = "nccl",
    enable_lora: bool = False,
    max_lora_rank: int = 16,
    max_loras: int = 1,
    max_model_len: int = 1024,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_LOGGING_LEVEL"] = logging_level
    if weight_transfer_backend == "ipc":
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if enable_lora:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        "1",
        "--weight-transfer-config",
        json.dumps({"backend": weight_transfer_backend}),
        "--load-format",
        load_format,
        "--max_model_len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if max_num_batched_tokens is not None:
        command.extend(
            ["--max-num-batched-tokens", str(max_num_batched_tokens)]
        )
    if enable_lora:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--max-loras",
                str(max_loras),
            ]
        )
    logger.info("Starting vLLM server: %s", " ".join(command))
    return subprocess.Popen(command, env=env, start_new_session=True)


def wait_for_server(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def stop_server(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def generate_completions(
    vllm_base_url: str,
    model_id: str,
    prompts: list[str],
    sampling_params: dict,
    batch_size: int | None = None,
) -> list[VLLMCompletion]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    prompt_batches = [prompts]
    if batch_size is not None:
        prompt_batches = [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]

    completions = []
    for prompt_batch in prompt_batches:
        payload = {
            "model": model_id,
            "prompt": prompt_batch,
            "temperature": sampling_params["temperature"],
            "max_tokens": sampling_params["max_tokens"],
            "n": sampling_params["n"],
            "seed": sampling_params["seed"],
            "return_token_ids": True,
        }
        if sampling_params.get("stop") is not None:
            payload["stop"] = sampling_params["stop"]
            payload["include_stop_str_in_output"] = sampling_params.get("include_stop_str_in_output", False)

        response = _http_json("POST", f"{vllm_base_url}/v1/completions", payload, timeout=3600)
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        completions.extend(
            VLLMCompletion(
                text=choice["text"],
                token_ids=choice.get("token_ids") or [],
                finish_reason=choice.get("finish_reason"),
            )
            for choice in choices
        )
    return completions


def generate_confidence_predictions(
    vllm_base_url: str,
    model_id: str,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    group_size: int,
    batch_size: int,
    seed: int,
) -> VLLMConfidencePredictions:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(candidate_token_ids) != group_size + 1:
        raise ValueError("Expected one candidate token for each count from 0 to group_size")
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("candidate_token_ids must be unique")

    hard_confidences: list[float] = []
    expected_confidences: list[float] = []
    instances_with_missing_logprobs = 0
    prompt_list = list(prompts)
    started = time.perf_counter()
    for start in range(0, len(prompt_list), batch_size):
        prompt_batch = prompt_list[start : start + batch_size]
        payload = {
            "model": model_id,
            "prompt": prompt_batch,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 1,
            "n": 1,
            "seed": seed,
            "logprobs": len(candidate_token_ids),
            "allowed_token_ids": list(candidate_token_ids),
            "return_tokens_as_token_ids": True,
        }
        response = _http_json(
            "POST",
            f"{vllm_base_url}/v1/completions",
            payload,
            timeout=3600,
        )
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        if len(choices) != len(prompt_batch):
            raise RuntimeError(
                f"Expected {len(prompt_batch)} confidence choices, got {len(choices)}"
            )

        for choice in choices:
            logprobs = choice.get("logprobs")
            top_logprobs = None if logprobs is None else logprobs.get("top_logprobs")
            if not top_logprobs or top_logprobs[0] is None:
                raise RuntimeError("vLLM did not return confidence token log-probabilities")
            token_logprobs = top_logprobs[0]
            candidate_logprobs = [
                float(token_logprobs.get(f"token_id:{token_id}", -math.inf))
                for token_id in candidate_token_ids
            ]
            if any(not math.isfinite(value) for value in candidate_logprobs):
                instances_with_missing_logprobs += 1
            maximum = max(candidate_logprobs)
            weights = [
                math.exp(value - maximum) if math.isfinite(value) else 0.0
                for value in candidate_logprobs
            ]
            normalizer = sum(weights)
            if normalizer == 0.0:
                raise RuntimeError("No candidate confidence token had finite probability")
            probabilities = [weight / normalizer for weight in weights]
            predicted_count = max(
                range(len(probabilities)),
                key=probabilities.__getitem__,
            )
            expected_count = sum(
                count * probability
                for count, probability in enumerate(probabilities)
            )
            hard_confidences.append(predicted_count / group_size)
            expected_confidences.append(expected_count / group_size)

    return VLLMConfidencePredictions(
        hard_confidences=hard_confidences,
        expected_confidences=expected_confidences,
        metrics={
            "instances_with_missing_logprobs": float(
                instances_with_missing_logprobs
            ),
            "inference_seconds": time.perf_counter() - started,
        },
    )


def init_weight_sync(vllm_base_url: str, policy_device: str):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    inference_world_size = _http_json("GET", f"{vllm_base_url}/get_world_size", timeout=10)["world_size"]
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()
    init_info = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": world_size,
    }

    torch.cuda.set_device(torch.device(policy_device))
    with ThreadPoolExecutor(max_workers=1) as executor:
        init_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/init_weight_transfer_engine",
            {"init_info": init_info},
            60,
        )
        weight_sync_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_future.result()

    return weight_sync_group


def sync_policy_weights(policy: torch.nn.Module, vllm_base_url: str, weight_sync_group) -> None:
    """Copy policy weights into vLLM and invalidate caches derived from old weights."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    weights = list(policy.named_parameters())
    update_info = {
        "names": [name for name, _ in weights],
        "dtype_names": [str(tensor.dtype).split(".")[-1] for _, tensor in weights],
        "shapes": [list(tensor.shape) for _, tensor in weights],
        "packed": True,
    }

    torch.cuda.set_device(next(policy.parameters()).device)
    _http_json("POST", f"{vllm_base_url}/pause", timeout=60)
    with ThreadPoolExecutor(max_workers=1) as executor:
        update_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/update_weights",
            {"update_info": update_info},
            300,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(weights),
            trainer_args=NCCLTrainerSendWeightsArgs(
                group=weight_sync_group,
                packed=True,
            ),
        )
        update_future.result()
    _http_json("POST", f"{vllm_base_url}/reset_prefix_cache", timeout=60)
    _http_json("POST", f"{vllm_base_url}/resume", timeout=60)


def sync_policy_weights_ipc(policy: torch.nn.Module, vllm_base_url: str) -> None:
    """Copy policy weights into a same-GPU vLLM worker via CUDA IPC handles."""
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCTrainerSendWeightsArgs,
        IPCWeightTransferEngine,
    )

    torch.cuda.set_device(next(policy.parameters()).device)
    _http_json("POST", f"{vllm_base_url}/pause", timeout=60)
    try:
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=iter(policy.named_parameters()),
            trainer_args=IPCTrainerSendWeightsArgs(mode="http", url=vllm_base_url),
        )
        _http_json("POST", f"{vllm_base_url}/reset_prefix_cache", timeout=60)
    finally:
        _http_json("POST", f"{vllm_base_url}/resume", timeout=60)
