#!/usr/bin/env python3
"""
VASU — Synthetic Tool Call Data Generator
Uses DeepSeek-R1-Distill-Llama-70B (GGUF on droplet) to generate 1,000 tool call examples.
Covers Vasu's 7 tools with Hinglish queries mapping to structured JSON tool calls.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Reuse the teacher model loading from gen_hinglish_conversations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="[GEN-TOOLS %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/gen_tool_calls.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/teacher_model"
OUTPUT_FILE = f"{SCRATCH}/datasets/synthetic/tool_call_data.json"
TARGET_COUNT = 1000
BATCH_SIZE = 10

TEACHER_GGUF_REPO = "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"
TEACHER_GGUF_FILE = "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

VASU_TOOLS_SCHEMA = """Available tools for Vasu:
1. invoke_camera(query: str, camera: "front"|"rear") — Take photo and analyze with VLM
2. set_alarm(time: "HH:MM", label: str, days: list) — Set alarm/reminder
3. add_note(content: str, title: str) — Save a note
4. web_search(query: str) — Search the web
5. toggle_device(device: str, state: "on"|"off") — Control smart home device
6. get_time() — Get current time
7. get_weather(location: str) — Get weather info"""

SYSTEM_PROMPT = """You are generating training data for a home AI assistant named Vasu.
Generate examples where a user asks something and Vasu either:
A) Directly answers (no tool needed), OR
B) Calls a tool with proper JSON format

{tools_schema}

Tool call format Vasu should use:
```json
{{"tool": "<tool_name>", "params": {{...}}}}
```

Rules:
- Queries are in Hinglish (Hindi-English mix), formal Hindi, or English
- Vasu responds SHORT — max 2 sentences + tool call if needed
- Include the COMPLETE conversation: user query → Vasu response (with or without tool call)
- For ambiguous requests, Vasu asks ONE clarifying question first
- Mix of: tool calls, direct answers, clarification-then-tool-call sequences

Output a JSON array of {batch_size} conversation objects, each with "messages" key.
Focus on: {topic}"""

TOOL_TOPICS = [
    "camera usage — checking rooms, reading labels, identifying objects, taking photos",
    "alarm and reminder setting — morning alarms, medication reminders, meeting timers",
    "note-taking — shopping lists, to-do items, quick thoughts, diary entries",
    "web search — looking up information, news, recipes, how-to guides",
    "smart home control — lights, fans, AC, TV, various devices on/off",
    "time queries — what time is it, timezone questions, time calculations",
    "weather queries — today's weather, rain forecast, temperature, weekly forecast",
    "multi-step tasks — set alarm AND add note, check weather AND suggest outfit",
    "ambiguous requests requiring clarification before tool use",
    "requests that seem like tool calls but Vasu can answer directly",
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
            repo_id=TEACHER_GGUF_REPO,
            filename=TEACHER_GGUF_FILE,
            local_dir=MODEL_DIR,
            token=hf_token,
        )
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def load_model():
    """Load teacher model — try llama.cpp first, then transformers."""
    model_path = download_teacher_if_needed()

    if model_path and os.path.exists(model_path):
        try:
            from llama_cpp import Llama
            log.info(f"Loading via llama.cpp: {model_path}")
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_batch=512,
                verbose=False,
            )
            return llm, "llamacpp"
        except Exception as e:
            log.warning(f"llama.cpp failed: {e}")

    # Fallback: transformers
    log.info("Loading via transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    return (model, tokenizer), "transformers"


def generate(llm, engine: str, topic: str, batch_size: int) -> str:
    """Generate text using the loaded model."""
    prompt = SYSTEM_PROMPT.format(
        tools_schema=VASU_TOOLS_SCHEMA,
        batch_size=batch_size,
        topic=topic,
    )

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


def parse_and_validate(text: str) -> list:
    """Parse and validate generated tool call conversations."""
    import re

    text = text.strip()

    # Try to extract JSON array
    for strategy in [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find('['):t.rfind(']') + 1]),
    ]:
        try:
            data = strategy(text)
            if isinstance(data, list):
                valid = []
                for conv in data:
                    messages = conv.get("messages", conv.get("conversations", []))
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
    log.info("║  VASU — TOOL CALL DATA GENERATOR             ║")
    log.info("╚══════════════════════════════════════════════╝")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Resume
    all_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                all_data = json.load(f)
            log.info(f"Resuming from {len(all_data)} existing examples")
        except Exception:
            all_data = []

    if len(all_data) >= TARGET_COUNT:
        log.info("Target met.")
        return

    llm, engine = load_model()
    log.info(f"Engine: {engine}")

    topic_idx = 0
    while len(all_data) < TARGET_COUNT:
        topic = TOOL_TOPICS[topic_idx % len(TOOL_TOPICS)]
        remaining = min(BATCH_SIZE, TARGET_COUNT - len(all_data))

        log.info(f"Generating: topic='{topic}', progress={len(all_data)}/{TARGET_COUNT}")

        try:
            text = generate(llm, engine, topic, remaining)
            valid = parse_and_validate(text)

            if valid:
                all_data.extend(valid)
                log.info(f"  → {len(valid)} valid examples ({len(all_data)}/{TARGET_COUNT})")

            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"Error: {e}")

        topic_idx += 1
        time.sleep(0.5)

    log.info(f"✓ Generated {len(all_data)} tool call examples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
