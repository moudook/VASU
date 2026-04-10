#!/usr/bin/env python3
"""
VASU — VLM Data Preprocessor
Formats vision datasets for SmolVLM fine-tuning.
Creates image-text pairs in the format expected by SmolVLM trainer.
"""

import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[PREPROCESS-VLM %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/preprocess_vlm.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASETS_DIR = "/scratch/vasu/datasets/vlm"
SYNTHETIC_DIR = "/scratch/vasu/datasets/synthetic"
OUTPUT_DIR = "/scratch/vasu/datasets/vlm/processed"
IMAGE_DIR = f"{OUTPUT_DIR}/images"


def safe_load(path: str):
    """Load dataset from disk."""
    from datasets import load_from_disk
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning(f"Dataset not found: {path}")
        return None
    try:
        return load_from_disk(path)
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None


def process_cauldron_subset(name: str, path: str):
    """Process a Cauldron subset (COCO-QA, TextVQA, DocVQA, AI2D)."""
    ds = safe_load(path)
    if ds is None:
        return []

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    entries = []
    for idx, example in enumerate(ds):
        try:
            images = example.get("images", [])
            texts = example.get("texts", [])

            if not texts:
                continue

            # Extract Q&A from texts
            for text_item in texts:
                if isinstance(text_item, dict):
                    user_msg = text_item.get("user", "")
                    assistant_msg = text_item.get("assistant", "")
                elif isinstance(text_item, str):
                    continue
                else:
                    continue

                if user_msg and assistant_msg:
                    entry = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": user_msg},
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": assistant_msg},
                                ],
                            },
                        ],
                    }
                    # Store image reference
                    if images:
                        entry["image"] = images[0] if isinstance(images, list) else images
                    entries.append(entry)
        except Exception as e:
            continue

    log.info(f"  {name}: {len(entries)} entries")
    return entries


def process_llava_instruct():
    """Process LLaVA-Instruct-150K."""
    ds = safe_load(f"{DATASETS_DIR}/llava_instruct")
    if ds is None:
        return []

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        ds = ds[splits[0]]

    entries = []
    for example in ds:
        try:
            conversations = example.get("conversations", [])
            image = example.get("image", None)

            if not conversations or len(conversations) < 2:
                continue

            messages = []
            for i, turn in enumerate(conversations[:6]):
                role = "user" if turn.get("from") in ("human", "user") else "assistant"
                value = turn.get("value", "")

                if i == 0 and role == "user":
                    # First user message includes image
                    content = [
                        {"type": "image"},
                        {"type": "text", "text": value.replace("<image>\n", "").replace("<image>", "")},
                    ]
                else:
                    content = [{"type": "text", "text": value}]

                messages.append({"role": role, "content": content})

            entry = {"messages": messages}
            if image:
                entry["image"] = image
            entries.append(entry)
        except Exception:
            continue

    log.info(f"  llava_instruct: {len(entries)} entries")
    return entries


def process_synthetic_vlm():
    """Load synthetic VLM home scene data."""
    path = f"{SYNTHETIC_DIR}/vlm_home_data.json"
    if not os.path.exists(path):
        log.warning("Synthetic VLM data not found")
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        log.info(f"  synthetic_vlm: {len(data)} entries")
        return data
    except Exception as e:
        log.error(f"Failed to load synthetic VLM data: {e}")
        return []


def main():
    log.info("╔══════════════════════════════════════╗")
    log.info("║   VASU VLM DATA PREPROCESSING        ║")
    log.info("╚══════════════════════════════════════╝")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    all_entries = []

    # Process Cauldron subsets
    cauldron_subsets = {
        "cauldron_coco": f"{DATASETS_DIR}/cauldron_coco",
        "cauldron_textvqa": f"{DATASETS_DIR}/cauldron_textvqa",
        "cauldron_docvqa": f"{DATASETS_DIR}/cauldron_docvqa",
        "cauldron_ai2d": f"{DATASETS_DIR}/cauldron_ai2d",
    }

    for name, path in cauldron_subsets.items():
        entries = process_cauldron_subset(name, path)
        all_entries.extend(entries)

    # Process LLaVA
    llava = process_llava_instruct()
    all_entries.extend(llava)

    # Process synthetic data
    synthetic = process_synthetic_vlm()
    all_entries.extend(synthetic)

    if not all_entries:
        log.error("No VLM training data available!")
        return

    # Save as JSON lines for training
    output_path = f"{OUTPUT_DIR}/train.jsonl"
    with open(output_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"✓ Saved {len(all_entries)} entries → {output_path}")

    # Also save as HF dataset
    from datasets import Dataset
    # Flatten for HF Dataset (store messages as JSON string)
    flat = []
    for entry in all_entries:
        flat.append({
            "messages_json": json.dumps(entry.get("messages", []), ensure_ascii=False),
            "image_ref": str(entry.get("image", "")),
        })

    ds = Dataset.from_list(flat)
    ds = ds.shuffle(seed=42)
    ds.save_to_disk(f"{OUTPUT_DIR}/train_dataset")
    log.info(f"✓ HF Dataset saved → {OUTPUT_DIR}/train_dataset")

    log.info(f"VLM preprocessing complete. Total: {len(all_entries)} entries")


if __name__ == "__main__":
    main()
