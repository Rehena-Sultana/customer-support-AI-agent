"""Load the base Qwen model and attach the user's saved PEFT/LoRA adapter (optional)."""
from functools import lru_cache
from pathlib import Path
import json
import re

from .config import BASE_MODEL, ADAPTER_DIR, DISABLE_MODEL, USE_ADAPTER

SYSTEM_PROMPT = (
    "You are a helpful, professional customer-support assistant for our company website live chat. "
    "Your primary goal is to answer the customer's question directly, politely, and naturally based strictly on the provided company knowledge excerpts. "
    "Never mention Twitter, DMs, Direct Messages, social media, or signature initials (such as /CB). "
    "Do not invent facts, policies, or account details. "
    "If the knowledge states that refunds or account changes cannot be authorized in chat, explain that politely and offer to connect them with human support."
)


def _adapter_exists(path: str) -> bool:
    p = Path(path)
    return p.exists() and (p / "adapter_config.json").exists()


@lru_cache(maxsize=2)
def load_model(use_adapter: bool | None = None):
    import os
    import src.config as cfg
    if getattr(cfg, "DISABLE_MODEL", False) or os.getenv("SUPPORT_DISABLE_MODEL", "0").strip().lower() in {"1", "true", "yes"}:
        raise RuntimeError("SUPPORT_DISABLE_MODEL is enabled")

    adapter_active = (getattr(cfg, "USE_ADAPTER", True) and os.getenv("SUPPORT_USE_ADAPTER", "1").strip().lower() in {"1", "true", "yes"}) if use_adapter is None else use_adapter

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(getattr(cfg, "ADAPTER_DIR", ADAPTER_DIR)).expanduser() if getattr(cfg, "ADAPTER_DIR", ADAPTER_DIR) else None
    has_adapter = adapter_active and adapter_path and _adapter_exists(str(adapter_path))

    tokenizer_source = str(adapter_path) if (has_adapter and (adapter_path / "tokenizer_config.json").exists()) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=quant_config, device_map="auto"
            )
        except Exception as exc:
            raise RuntimeError(
                "GPU/4-bit model loading failed. Check the installed torch, transformers, "
                "accelerate, and bitsandbytes versions. Original error: " + str(exc)
            ) from exc
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    if has_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base

    model.eval()
    return tokenizer, model


def generate_response(message: str, context: str = "", tool_result: str = "", max_new_tokens: int = 180) -> str:
    """Generate a customer-facing response using the model and retrieved knowledge."""
    tokenizer, model = load_model()
    evidence = []
    if context.strip():
        evidence.append("Company Knowledge Excerpts:\n" + context.strip())
    if tool_result.strip():
        evidence.append("Verified Result From Support System:\n" + tool_result.strip())

    if evidence:
        user_text = (
            "\n\n".join(evidence)
            + "\n\nCustomer Question:\n"
            + message.strip()
            + "\n\nPlease provide a clear, natural, and helpful answer based strictly on the knowledge above:"
        )
    else:
        user_text = message.strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up Twitter deflection artifacts if present
    is_twitter_canned = (
        bool(re.search(r"\b(dm|dms|direct message|direct messages)\b", answer, re.IGNORECASE))
        or "send us a note" in answer.lower()
        or "here to help. please" in answer.lower()
        or "here to help! please" in answer.lower()
        or bool(re.search(r"\s/[A-Za-z]{2}\b", answer))
    )
    if is_twitter_canned:
        if tool_result.strip():
            answer = f"I've processed your request: {tool_result.strip()}"
        elif context.strip():
            first_section = context.strip().split("\n\n")[0].replace("#", "").strip()
            answer = (
                f"Based on our company policy: {first_section}\n\n"
                "If you would like to proceed with this request, please let me know and I can connect you with an authorized human representative."
            )
        else:
            answer = "I have logged your request. Please let me know if you would like any additional assistance."

    return answer or "I couldn't prepare a reliable answer. I can connect you with a human support representative."


def try_generate_response(message: str, context: str = "", tool_result: str = "") -> str | None:
    """Use the model if configured; return None when unavailable so the app can fall back."""
    try:
        return generate_response(message, context=context, tool_result=tool_result)
    except Exception:
        return None
