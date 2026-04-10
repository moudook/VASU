#!/usr/bin/env python3
"""
VASU — Reasoning Traces Generator
Uses DeepSeek-R1-Distill-Llama-70B (GGUF on droplet) to generate 2,000 think/no-think
pairs that teach Vasu WHEN to think and HOW MUCH.
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[GEN-REASONING %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/gen_reasoning.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/teacher_model"
OUTPUT_FILE = f"{SCRATCH}/datasets/synthetic/reasoning_traces.json"
TARGET_COUNT = 2000
BATCH_SIZE = 5

TEACHER_GGUF_REPO = "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"
TEACHER_GGUF_FILE = "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

SYSTEM_PROMPT_THINKING = """Generate {batch_size} question-answer pairs for a home AI assistant named Vasu.
These are HARD questions that require reasoning. Vasu should use a <think> block before answering.

Rules:
- The <think> block must be CONCISE — maximum 150 tokens. Not stream-of-consciousness.
- After thinking, give a SHORT direct answer (1-3 sentences).
- Questions can be in Hinglish, Hindi, or English.
- Topics: {topic}

Format each as a JSON object with "messages" array:
[{{"role": "user", "content": "question"}}, {{"role": "assistant", "content": "<think>\\nbrief reasoning\\n</think>\\n\\nDirect answer here."}}]

Output a JSON array of {batch_size} conversations."""

SYSTEM_PROMPT_DIRECT = """Generate {batch_size} question-answer pairs for a home AI assistant named Vasu.
These are SIMPLE questions. Vasu should answer DIRECTLY without any thinking.

Rules:
- NO <think> block. Answer immediately.
- Maximum 1-2 sentences.
- Questions can be in Hinglish, Hindi, or English.
- Topics: {topic}

Format: JSON array of objects with "messages" key.
[{{"role": "user", "content": "simple question"}}, {{"role": "assistant", "content": "direct short answer"}}]

Output a JSON array of {batch_size} conversations."""

HARD_TOPICS = [
    "multi-step math word problems",
    "logical puzzles and riddles in Hindi",
    "planning a complex task (trip, event, project)",
    "comparing pros and cons of 2+ options",
    "debugging a technical problem step by step",
    "explaining a complex scientific concept simply",
    "financial calculations and budgeting",
    "recipe scaling and cooking time adjustments",
    "scheduling conflicts and time management",
    "code debugging and programming logic",
]

SIMPLE_TOPICS = [
    "greetings and small talk",
    "simple factual questions (capitals, dates, definitions)",
    "time and day queries",
    "basic math (2+2, percentages of small numbers)",
    "yes/no questions about common knowledge",
    "simple translations between Hindi and English",
    "weather related casual questions",
    "simple device commands (light on/off)",
    "asking for a single word definition",
    "asking what day/date it is",
]


def download_teacher_if_needed():
    """Download teacher model if not already present."""
    gguf_path = os.path.join(MODEL_DIR, TEACHER_GGUF_FILE)
    if os.path.exists(gguf_path):
        return gguf_path

    os.makedirs(MODEL_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN")
    try:
        return hf_hub_download(
            repo_id=TEACHER_GGUF_REPO, filename=TEACHER_GGUF_FILE,
            local_dir=MODEL_DIR, token=hf_token,
        )
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def load_model():
    """Load the teacher model."""
    model_path = download_teacher_if_needed()

    if model_path and os.path.exists(model_path):
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1,
                        n_batch=512, verbose=False)
            return llm, "llamacpp"
        except Exception as e:
            log.warning(f"llama.cpp failed: {e}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    return (model, tokenizer), "transformers"


def generate(llm, engine: str, prompt: str) -> str:
    """Generate text."""
    if engine == "llamacpp":
        output = llm(prompt, max_tokens=3000, temperature=0.7, top_p=0.9)
        return output["choices"][0]["text"]
    else:
        import torch
        model, tokenizer = llm
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=3000, temperature=0.7, top_p=0.9, do_sample=True,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text: str) -> list:
    """Extract JSON array from text."""
    text = text.strip()
    for strategy in [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find('['):t.rfind(']') + 1]),
    ]:
        try:
            data = strategy(text)
            if isinstance(data, list):
                valid = []
                for conv in data:
                    messages = conv.get("messages", [])
                    if messages and len(messages) >= 2:
                        normalized = []
                        for msg in messages:
                            role = msg.get("role", "user")
                            if role in ("human",):
                                role = "user"
                            elif role in ("gpt", "bot"):
                                role = "assistant"
                            normalized.append({
                                "role": role,
                                "content": msg.get("content", "").strip(),
                            })
                        valid.append({"messages": normalized})
                return valid
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return []


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU — REASONING TRACES GENERATOR           ║")
    log.info("╚══════════════════════════════════════════════╝")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                all_data = json.load(f)
            log.info(f"Resuming from {len(all_data)}")
        except Exception:
            all_data = []

    if len(all_data) >= TARGET_COUNT:
        log.info("Target met.")
        return

    llm, engine = load_model()
    log.info(f"Engine: {engine}")

    topic_idx = 0
    # Alternate between hard (thinking) and simple (no-thinking) examples
    while len(all_data) < TARGET_COUNT:
        is_hard = (topic_idx % 2 == 0)

        if is_hard:
            topic = HARD_TOPICS[topic_idx // 2 % len(HARD_TOPICS)]
            prompt = SYSTEM_PROMPT_THINKING.format(batch_size=BATCH_SIZE, topic=topic)
            label = "HARD"
        else:
            topic = SIMPLE_TOPICS[topic_idx // 2 % len(SIMPLE_TOPICS)]
            prompt = SYSTEM_PROMPT_DIRECT.format(batch_size=BATCH_SIZE, topic=topic)
            label = "SIMPLE"

        log.info(f"Generating {label}: topic='{topic}', progress={len(all_data)}/{TARGET_COUNT}")

        try:
            text = generate(llm, engine, prompt)
            valid = parse_json(text)

            if valid:
                all_data.extend(valid)
                log.info(f"  → {len(valid)} valid ({len(all_data)}/{TARGET_COUNT})")

            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"Error: {e}")

        topic_idx += 1
        time.sleep(0.5)

    log.info(f"✓ Generated {len(all_data)} reasoning pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
