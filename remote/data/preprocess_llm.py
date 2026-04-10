#!/usr/bin/env python3
"""
VASU — LLM Data Preprocessor
Tokenizes and formats all LLM training data into chat-template format for Qwen3.
Produces ready-to-train Arrow datasets for each stage.
"""

import json
import logging
import os
import sys
from pathlib import Path

import datasets
from datasets import Dataset, concatenate_datasets, load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="[PREPROCESS-LLM %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/preprocess_llm.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASETS_DIR = "/scratch/vasu/datasets/llm"
OUTPUT_DIR = "/scratch/vasu/datasets/llm/processed"
SYNTHETIC_DIR = "/scratch/vasu/datasets/synthetic"

# Vasu system prompt
VASU_SYSTEM_PROMPT = """You are Vasu (वासु), a home AI assistant. Rules:
- Speak Hinglish naturally (Hindi-English code-switching)
- Match user's language: Hindi→Hindi, English→English, Hinglish→Hinglish
- Keep responses SHORT: 1-3 sentences max for simple queries
- Never use filler: no "Great question!", "Certainly!", "As an AI...", "I'd be happy to help"
- Answer directly. No unnecessary disclaimers.
- Ask ONE clarifying question before complex/irreversible actions
- For tool calls, output valid JSON with the tool schema
- Adapt tone: technical users get technical answers, family members get simpler explanations"""

# Vasu tool schema
VASU_TOOLS = [
    {"name": "invoke_camera", "params": {"query": "string", "camera": "front|rear"}},
    {"name": "set_alarm", "params": {"time": "HH:MM", "label": "string", "days": "array"}},
    {"name": "add_note", "params": {"content": "string", "title": "string"}},
    {"name": "web_search", "params": {"query": "string"}},
    {"name": "toggle_device", "params": {"device": "string", "state": "on|off"}},
    {"name": "get_time", "params": {}},
    {"name": "get_weather", "params": {"location": "string"}},
]


def safe_load(path: str):
    """Load a dataset from disk, return None if missing."""
    if not os.path.exists(path):
        log.warning(f"Dataset not found: {path}")
        return None
    marker = os.path.join(path, ".download_complete")
    if not os.path.exists(marker):
        log.warning(f"Dataset incomplete: {path}")
        return None
    try:
        return load_from_disk(path)
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None


def to_chat_format(system: str, conversations: list) -> dict:
    """Convert to Qwen3 chat template format."""
    messages = [{"role": "system", "content": system}]
    for turn in conversations:
        messages.append(turn)
    return {"messages": messages}


def process_ultrachat(ds):
    """Process UltraChat into chat format."""
    processed = []
    if ds is None:
        return processed

    # Handle DatasetDict
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        messages = example.get("messages", [])
        if not messages or len(messages) < 2:
            continue
        entry = to_chat_format(VASU_SYSTEM_PROMPT, messages[:10])  # Cap at 10 turns
        processed.append(entry)
    return processed


def process_airoboros(ds):
    """Process Airoboros — primary behavior template (direct, no refusals)."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        conversations = example.get("conversations", [])
        if not conversations:
            continue
        messages = []
        for turn in conversations:
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            messages.append({"role": role, "content": turn.get("value", "")})
        if messages:
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages[:10])
            processed.append(entry)
    return processed


def process_openhermes(ds):
    """Process OpenHermes-2.5."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        conversations = example.get("conversations", [])
        if not conversations:
            continue
        messages = []
        for turn in conversations:
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            messages.append({"role": role, "content": turn.get("value", "")})
        if messages:
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages[:10])
            processed.append(entry)
    return processed


def process_wizardlm(ds):
    """Process WizardLM evol instruct."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        if instruction and output:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
            processed.append(entry)
    return processed


def process_platypus(ds):
    """Process Open-Platypus."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        inp = example.get("input", "")
        if instruction and output:
            user_msg = f"{instruction}\n{inp}" if inp else instruction
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
            processed.append(entry)
    return processed


def process_metamath(ds):
    """Process MetaMathQA."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        query = example.get("query", "")
        response = example.get("response", "")
        if query and response:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
            processed.append(entry)
    return processed


def process_aya(ds):
    """Process Aya dataset — filter Hindi and English."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    for example in ds:
        lang = example.get("language", "")
        if lang not in ("hin", "eng", "hi", "en"):
            continue
        inputs = example.get("inputs", "")
        targets = example.get("targets", "")
        if inputs and targets:
            messages = [
                {"role": "user", "content": inputs},
                {"role": "assistant", "content": targets},
            ]
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
            processed.append(entry)
    return processed


def process_sangraha(ds):
    """Process Sangraha Hindi text — convert to instruction format."""
    processed = []
    if ds is None:
        return processed

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    count = 0
    for example in ds:
        text = example.get("text", "")
        if not text or len(text) < 50:
            continue
        # Create a summarization / discussion task from Hindi text
        messages = [
            {"role": "user", "content": f"Yeh text padho aur iska summary do:\n{text[:1500]}"},
            {"role": "assistant", "content": text[:500]},  # Use beginning as proxy summary
        ]
        entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
        processed.append(entry)
        count += 1
        if count >= 10000:  # Cap raw text dataset
            break
    return processed


def process_synthetic(path: str):
    """Load synthetic generated data (JSON format)."""
    processed = []
    if not os.path.exists(path):
        log.warning(f"Synthetic data not found: {path}")
        return processed
    try:
        with open(path) as f:
            data = json.load(f)
        for item in data:
            messages = item.get("messages", item.get("conversations", []))
            if messages:
                entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
                processed.append(entry)
    except Exception as e:
        log.error(f"Failed to load synthetic data {path}: {e}")
    return processed


def build_stage1_dataset():
    """Stage 1: Conversational SFT — all base datasets + synthetic Hinglish."""
    log.info("═══ Building Stage 1 dataset (Conversational SFT) ═══")

    all_data = []

    processors = [
        ("ultrachat", f"{DATASETS_DIR}/ultrachat", process_ultrachat),
        ("airoboros", f"{DATASETS_DIR}/airoboros", process_airoboros),
        ("openhermes", f"{DATASETS_DIR}/openhermes", process_openhermes),
        ("wizardlm", f"{DATASETS_DIR}/wizardlm", process_wizardlm),
        ("platypus", f"{DATASETS_DIR}/platypus", process_platypus),
        ("metamath", f"{DATASETS_DIR}/metamath", process_metamath),
        ("aya", f"{DATASETS_DIR}/aya", process_aya),
        ("sangraha", f"{DATASETS_DIR}/sangraha", process_sangraha),
    ]

    for name, path, processor in processors:
        ds = safe_load(path)
        data = processor(ds)
        log.info(f"  {name}: {len(data)} examples")
        all_data.extend(data)

    # Add synthetic Hinglish conversations
    synthetic = process_synthetic(f"{SYNTHETIC_DIR}/hinglish_conversations.json")
    log.info(f"  synthetic_hinglish: {len(synthetic)} examples")
    all_data.extend(synthetic)

    if not all_data:
        log.error("No data for Stage 1!")
        return

    # Convert to HF Dataset
    ds = Dataset.from_list(all_data)
    ds = ds.shuffle(seed=42)
    out_path = f"{OUTPUT_DIR}/stage1_sft"
    ds.save_to_disk(out_path)
    log.info(f"✓ Stage 1 dataset: {len(ds)} examples → {out_path}")


def build_stage2_dataset():
    """Stage 2: Tool calling SFT."""
    log.info("═══ Building Stage 2 dataset (Tool Calling) ═══")

    all_data = []

    tool_datasets = [
        ("hermes_fc", f"{DATASETS_DIR}/hermes_fc"),
        ("glaive_fc", f"{DATASETS_DIR}/glaive_fc"),
        ("xlam_fc", f"{DATASETS_DIR}/xlam_fc"),
    ]

    tool_system = VASU_SYSTEM_PROMPT + f"\n\nAvailable tools:\n{json.dumps(VASU_TOOLS, indent=2)}"

    for name, path in tool_datasets:
        ds = safe_load(path)
        if ds is None:
            continue
        if hasattr(ds, "keys"):
            splits = list(ds.keys())
            ds = ds[splits[0]]

        count = 0
        for example in ds:
            # Try various field names used by different tool-call datasets
            conv = (example.get("conversations") or example.get("messages")
                    or example.get("data") or [])

            if not conv:
                # Try instruction/output format
                inst = example.get("instruction", example.get("query", ""))
                out = example.get("output", example.get("answer", ""))
                if inst and out:
                    conv = [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": out},
                    ]

            if not conv:
                continue

            messages = []
            for turn in conv:
                role_raw = turn.get("role", turn.get("from", ""))
                if role_raw in ("human", "user"):
                    role = "user"
                elif role_raw in ("gpt", "assistant", "tool_call"):
                    role = "assistant"
                elif role_raw in ("system",):
                    role = "system"
                else:
                    role = "user"
                content = turn.get("content", turn.get("value", ""))
                messages.append({"role": role, "content": content})

            entry = to_chat_format(tool_system, messages[:10])
            all_data.append(entry)
            count += 1
        log.info(f"  {name}: {count} examples")

    # Add synthetic tool call data
    synthetic = process_synthetic(f"{SYNTHETIC_DIR}/tool_call_data.json")
    log.info(f"  synthetic_tools: {len(synthetic)} examples")
    all_data.extend(synthetic)

    if not all_data:
        log.warning("No data for Stage 2!")
        return

    ds = Dataset.from_list(all_data)
    ds = ds.shuffle(seed=42)
    out_path = f"{OUTPUT_DIR}/stage2_tools"
    ds.save_to_disk(out_path)
    log.info(f"✓ Stage 2 dataset: {len(ds)} examples → {out_path}")


def build_stage3_dataset():
    """Stage 3: Reasoning distillation."""
    log.info("═══ Building Stage 3 dataset (Reasoning) ═══")

    all_data = []

    reasoning_datasets = [
        ("openr1_math", f"{DATASETS_DIR}/openr1_math"),
        ("s1k", f"{DATASETS_DIR}/s1k"),
        ("stratos", f"{DATASETS_DIR}/stratos"),
    ]

    for name, path in reasoning_datasets:
        ds = safe_load(path)
        if ds is None:
            continue
        if hasattr(ds, "keys"):
            splits = list(ds.keys())
            ds = ds[splits[0]]

        count = 0
        for example in ds:
            # Handle various reasoning dataset formats
            question = (example.get("problem") or example.get("question")
                        or example.get("instruction") or example.get("input", ""))
            answer = (example.get("solution") or example.get("answer")
                      or example.get("output") or example.get("response", ""))

            if not question or not answer:
                continue

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            entry = to_chat_format(VASU_SYSTEM_PROMPT, messages)
            all_data.append(entry)
            count += 1
        log.info(f"  {name}: {count} examples")

    # Add synthetic reasoning traces
    synthetic = process_synthetic(f"{SYNTHETIC_DIR}/reasoning_traces.json")
    log.info(f"  synthetic_reasoning: {len(synthetic)} examples")
    all_data.extend(synthetic)

    if not all_data:
        log.warning("No data for Stage 3!")
        return

    ds = Dataset.from_list(all_data)
    ds = ds.shuffle(seed=42)
    out_path = f"{OUTPUT_DIR}/stage3_reasoning"
    ds.save_to_disk(out_path)
    log.info(f"✓ Stage 3 dataset: {len(ds)} examples → {out_path}")


def build_grpo_dataset():
    """Stage 4: GRPO data — needs reward annotations."""
    log.info("═══ Building Stage 4 dataset (GRPO) ═══")

    all_data = []

    # Combine all available data for GRPO with reward signals
    # Load the truthy DPO dataset for anti-sycophancy
    ds = safe_load(f"{DATASETS_DIR}/truthy_dpo")
    if ds is not None:
        if hasattr(ds, "keys"):
            splits = list(ds.keys())
            ds = ds[splits[0]]

        for example in ds:
            prompt = example.get("prompt", example.get("question", ""))
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")

            if prompt and chosen:
                all_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected if rejected else "",
                })
        log.info(f"  truthy_dpo: {len(all_data)} examples")

    if not all_data:
        log.warning("No data for GRPO stage!")
        return

    ds = Dataset.from_list(all_data)
    ds = ds.shuffle(seed=42)
    out_path = f"{OUTPUT_DIR}/stage4_grpo"
    ds.save_to_disk(out_path)
    log.info(f"✓ Stage 4 dataset: {len(ds)} examples → {out_path}")


def main():
    log.info("╔══════════════════════════════════════╗")
    log.info("║   VASU LLM DATA PREPROCESSING        ║")
    log.info("╚══════════════════════════════════════╝")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    build_stage1_dataset()
    build_stage2_dataset()
    build_stage3_dataset()
    build_grpo_dataset()

    log.info("LLM preprocessing complete.")


if __name__ == "__main__":
    main()
