"""Load the base Qwen model and attach the user's saved PEFT/LoRA adapter."""
from functools import lru_cache
from pathlib import Path
import json

from .config import BASE_MODEL, ADAPTER_DIR, DISABLE_MODEL

SYSTEM_PROMPT = (
    "You are a careful customer-support assistant. Be polite and concise. "
    "Never invent account details, ticket status, refunds, or company policy. "
    "Use only the provided knowledge or verified tool results for factual claims. "
    "If information is missing, ask a question or recommend human support."
)


def _adapter_exists(path: str) -> bool:
    p = Path(path)
    return p.exists() and (p / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def load_model():
    """Returns (tokenizer, model); model loads once per Streamlit process."""
    if DISABLE_MODEL:
        raise RuntimeError("SUPPORT_DISABLE_MODEL is enabled")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(ADAPTER_DIR).expanduser()
    if not _adapter_exists(str(adapter_path)):
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. Set SUPPORT_ADAPTER_DIR in .env "
            "to the folder containing adapter_config.json and adapter weights."
        )

    # Adapter folders do not always include tokenizer files. In that case use base tokenizer.
    tokenizer_source = str(adapter_path) if (adapter_path / "tokenizer_config.json").exists() else BASE_MODEL
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
            # Provide a useful error rather than silently loading a huge model twice.
            raise RuntimeError(
                "GPU/4-bit model loading failed. Check the installed torch, transformers, "
                "accelerate, and bitsandbytes versions. Original error: " + str(exc)
            ) from exc
    else:
        # CPU mode is slower and uses more RAM, but allows basic local testing.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return tokenizer, model


def generate_response(message: str, context: str = "", tool_result: str = "", max_new_tokens: int = 180) -> str:
    """Generate a customer-facing response using the fine-tuned adapter."""
    tokenizer, model = load_model()
    evidence = []
    if context.strip():
        evidence.append("Retrieved company knowledge (use only if relevant):\n" + context.strip())
    if tool_result.strip():
        evidence.append("Verified result from a support tool:\n" + tool_result.strip())
    user_text = message.strip()
    if evidence:
        user_text = "\n\n".join(evidence) + "\n\nCustomer message:\n" + user_text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
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
    return answer or "I couldn't prepare a reliable answer. I can connect you with a human support representative."


def try_generate_response(message: str, context: str = "", tool_result: str = "") -> str | None:
    """Use the adapter if configured; return None when unavailable so the app can fall back."""
    try:
        return generate_response(message, context=context, tool_result=tool_result)
    except Exception:
        return None
