from importlib.util import find_spec

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_attn_implementation(device: str) -> str:
    device_str = str(device)
    if device_str == "cpu" or device_str.startswith("cpu:"):
        return "eager"
    if find_spec("flash_attn") is not None:
        return "flash_attention_2"
    if device_str.startswith("cuda"):
        return "sdpa"
    return "eager"


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation=_get_attn_implementation(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
