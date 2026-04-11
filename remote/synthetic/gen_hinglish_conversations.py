#!/usr/bin/env python3
"""
VASU — Synthetic Hinglish Conversation Generator
Uses DeepSeek-R1-Distill-Llama-70B (GGUF quantized) running locally on the GPU droplet
via vLLM to generate 3,000 Hinglish conversation pairs for Vasu SFT training.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[GEN-HINGLISH %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/gen_hinglish.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/teacher_model"
OUTPUT_FILE = f"{SCRATCH}/datasets/synthetic/hinglish_conversations.json"
TARGET_COUNT = 3000
BATCH_SIZE = 10  # conversations per generation call

# Teacher model config
TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
TEACHER_GGUF_REPO = "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"
TEACHER_GGUF_FILE = "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are a synthetic data generator. Generate realistic multi-turn conversations
between a user and a home AI assistant named Vasu (वासु). Follow these rules strictly:

1. Mix Hindi and English naturally mid-sentence (Hinglish). Example: "Bhai weather kaisa hai aaj?"
2. Vasu's responses MUST be SHORT — maximum 2 sentences.
3. Vasu NEVER uses filler phrases like "Great question!", "Certainly!", "As an AI...", "I'd be happy to help"
4. Vasu asks ONE clarifying question before complex/irreversible tasks.
5. Include variety:
   - Simple factual questions (capital of a country, math)
   - Smart home control ("light on kar", "alarm set kar")
   - Knowledge questions
   - Hindi-only requests from elderly users (formal Hindi, simpler language)
   - English-only from tech-savvy users
   - Mixed Hinglish from casual users
6. Some conversations should be from elderly family members in more formal Hindi.
7. Vasu adapts language/tone to match the user.

Output ONLY a valid JSON array of conversation objects. Each object has a "messages" key
containing an array of {role, content} objects. Include 2-6 turns per conversation.

Generate exactly {batch_size} different conversations covering diverse topics."""

TOPIC_CATEGORIES = [
    "daily life questions, weather, time, reminders",
    "cooking recipes, food, kitchen help",
    "technology troubleshooting, wifi, phone issues",
    "health and fitness basic advice",
    "news and current events discussion",
    "smart home control — lights, fans, AC, alarms",
    "education help — math, science, history",
    "entertainment — movies, songs, recommendations",
    "shopping lists and household management",
    "travel planning and directions",
    "formal Hindi elderly user — medicine reminders, family calls",
    "formal Hindi — religious/cultural queries, festivals",
    "English-only — programming, tech, coding questions",
    "security, privacy, technical concepts explained directly",
    "sensitive topics answered honestly without lectures",
]


def download_teacher_model():
    """Download the quantized DeepSeek-R1-Distill-Llama-70B GGUF to the droplet."""
    gguf_path = os.path.join(MODEL_DIR, TEACHER_GGUF_FILE)
    if os.path.exists(gguf_path):
        log.info(f"Teacher model already downloaded: {gguf_path}")
        return gguf_path

    log.info(f"Downloading teacher model: {TEACHER_GGUF_REPO}/{TEACHER_GGUF_FILE}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    try:
        downloaded = hf_hub_download(
            repo_id=TEACHER_GGUF_REPO,
            filename=TEACHER_GGUF_FILE,
            local_dir=MODEL_DIR,
            token=hf_token,
        )
        log.info(f"✓ Teacher model downloaded: {downloaded}")
        return downloaded
    except Exception as e:
        log.error(f"Failed to download GGUF: {e}")
        log.info("Trying alternative: loading via transformers + auto-GPTQ...")
        return None


def load_teacher_vllm(model_path: str):
    """Load teacher model using vLLM for fast batched generation."""
    try:
        from vllm import LLM, SamplingParams
        log.info("Loading teacher model via vLLM...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Single MI300X GPU
            max_model_len=8192,
            gpu_memory_utilization=0.90,  # Use 90% of 192GB VRAM
            trust_remote_code=True,
            dtype="bfloat16",
        )
        return llm, "vllm"
    except Exception as e:
        log.warning(f"vLLM failed: {e}. Falling back to llama-cpp-python.")
        return None, None


def load_teacher_llamacpp(model_path: str):
    """Load teacher model using llama.cpp Python bindings."""
    try:
        from llama_cpp import Llama
        log.info(f"Loading teacher model via llama.cpp: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # offload all layers to GPU
            n_batch=512,
            verbose=False,
        )
        return llm, "llamacpp"
    except ImportError:
        log.warning("llama-cpp-python not installed. Installing...")
        os.system("pip install llama-cpp-python --no-cache-dir")
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_batch=512,
            verbose=False,
        )
        return llm, "llamacpp"


def load_teacher_transformers():
    """Fallback: load teacher model via transformers (slower but always works)."""
    log.info(f"Loading teacher model via transformers: {TEACHER_MODEL_ID}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return (model, tokenizer), "transformers"


def generate_batch_vllm(llm, topic: str, batch_size: int):
    """Generate conversations using vLLM."""
    from vllm import SamplingParams

    prompt = SYSTEM_PROMPT.format(batch_size=batch_size) + f"\n\nFocus this batch on: {topic}"

    params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=3000,
        stop=None,
    )

    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text
    return text


def generate_batch_llamacpp(llm, topic: str, batch_size: int):
    """Generate conversations using llama.cpp."""
    prompt = SYSTEM_PROMPT.format(batch_size=batch_size) + f"\n\nFocus this batch on: {topic}"

    output = llm(
        prompt,
        max_tokens=3000,
        temperature=0.8,
        top_p=0.95,
        stop=None,
    )
    return output["choices"][0]["text"]


def generate_batch_transformers(model_tuple, topic: str, batch_size: int):
    """Generate conversations using transformers (fallback)."""
    import torch
    model, tokenizer = model_tuple

    prompt = SYSTEM_PROMPT.format(batch_size=batch_size) + f"\n\nFocus this batch on: {topic}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3000,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def parse_conversations(text: str) -> list:
    """Parse generated JSON text into conversation list."""
    # Try to find JSON array in the output
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "conversations" in data:
            return data["conversations"]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from mixed text
    import re
    matches = re.findall(r'\[[\s\S]*?\](?=\s*$|\s*\n\n)', text)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue

    # Last resort: find first [ to last ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    log.warning("Failed to parse JSON from generation output")
    return []


def validate_conversation(conv: dict) -> bool:
    """Validate a single conversation entry."""
    messages = conv.get("messages", conv.get("conversations", []))
    if not messages or len(messages) < 2:
        return False

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("user", "assistant", "system"):
            return False
        if not msg["content"] or len(msg["content"].strip()) < 2:
            return False

    return True


def normalize_conversation(conv: dict) -> dict:
    """Normalize conversation format."""
    messages = conv.get("messages", conv.get("conversations", []))
    normalized = []
    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        if role in ("human",):
            role = "user"
        elif role in ("gpt", "bot"):
            role = "assistant"
        content = msg.get("content", msg.get("value", ""))
        normalized.append({"role": role, "content": content.strip()})
    return {"messages": normalized}


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU — HINGLISH CONVERSATION GENERATOR      ║")
    log.info("╚══════════════════════════════════════════════╝")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load existing progress
    all_conversations = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                all_conversations = json.load(f)
            log.info(f"Resuming from {len(all_conversations)} existing conversations")
        except Exception:
            all_conversations = []

    if len(all_conversations) >= TARGET_COUNT:
        log.info(f"Already have {len(all_conversations)} conversations. Target met.")
        return

    # Try loading teacher model in priority order
    model_path = download_teacher_model()

    llm = None
    engine = None

    if model_path and os.path.exists(model_path):
        # Try vLLM first (fastest for batched generation)
        llm, engine = load_teacher_vllm(TEACHER_MODEL_ID)

        if llm is None:
            # Try llama.cpp (works with GGUF)
            llm, engine = load_teacher_llamacpp(model_path)

    if llm is None:
        # Fallback to transformers
        llm, engine = load_teacher_transformers()

    log.info(f"Using engine: {engine}")

    generate_fns = {
        "vllm": generate_batch_vllm,
        "llamacpp": generate_batch_llamacpp,
        "transformers": generate_batch_transformers,
    }
    
    if engine not in generate_fns:
        log.error(f"Unknown engine: {engine}")
        sys.exit(1)
        
    generate_fn = generate_fns[engine]

    # Generate conversations
    topic_idx = 0
    retries = 0
    max_retries = 3

    while len(all_conversations) < TARGET_COUNT:
        topic = TOPIC_CATEGORIES[topic_idx % len(TOPIC_CATEGORIES)]
        remaining = TARGET_COUNT - len(all_conversations)
        batch = min(BATCH_SIZE, remaining)

        log.info(f"Generating batch: topic='{topic}', batch={batch}, "
                 f"progress={len(all_conversations)}/{TARGET_COUNT}")

        try:
            text = generate_fn(llm, topic, batch)
            conversations = parse_conversations(text)

            valid = []
            for conv in conversations:
                if validate_conversation(conv):
                    valid.append(normalize_conversation(conv))

            if valid:
                all_conversations.extend(valid)
                retries = 0
                log.info(f"  → Got {len(valid)} valid conversations "
                         f"({len(all_conversations)}/{TARGET_COUNT})")
            else:
                retries += 1
                log.warning(f"  → No valid conversations in batch (retry {retries})")

            # Save progress every batch
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_conversations, f, ensure_ascii=False, indent=2)

        except Exception as e:
            retries += 1
            log.error(f"Generation error: {e}")
            if retries >= max_retries:
                log.error("Too many retries on same topic. Advancing.")
                retries = 0

        topic_idx += 1
        time.sleep(1)  # Brief pause between batches

    log.info(f"✓ Generated {len(all_conversations)} Hinglish conversations → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
