#!/usr/bin/env python3
"""
VASU — VLM Training: SmolVLM-500M Home Environment Fine-tune
Full fine-tune of SmolVLM-500M-Instruct on home scene VQA data.
Distributed across 8x MI300X GPUs.
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-VLM %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_vlm.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/vlm/processed"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/vlm"
OUTPUT_DIR = f"{SCRATCH}/models/vlm"

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def train():
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        TrainingArguments,
        Trainer,
    )
    from PIL import Image
    import io

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU VLM — SmolVLM-500M FINE-TUNE            ║")
    log.info("╚══════════════════════════════════════════════╝")

    num_gpus = torch.cuda.device_count()
    log.info(f"GPUs: {num_gpus}")

    hf_token = os.environ.get("HF_TOKEN")

    # Load processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=hf_token, trust_remote_code=True)

    # Load model — full fine-tune (500M is small enough)
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True,
    )

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load dataset
    train_path = f"{DATA_DIR}/train.jsonl"
    if not os.path.exists(train_path):
        # Try HF dataset format
        train_path_hf = f"{DATA_DIR}/train_dataset"
        if os.path.exists(train_path_hf):
            dataset = load_from_disk(train_path_hf)
        else:
            log.error("VLM training data not found!")
            sys.exit(1)
    else:
        # Load from JSONL
        entries = []
        with open(train_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        log.info(f"Loaded {len(entries)} entries from JSONL")

        from datasets import Dataset
        flat = []
        for entry in entries:
            flat.append({
                "messages_json": json.dumps(entry.get("messages", []), ensure_ascii=False),
                "image_ref": str(entry.get("image", "")),
            })
        dataset = Dataset.from_list(flat)

    log.info(f"Dataset: {len(dataset)} entries")

    # Create a dummy white image for entries without images
    dummy_image = Image.new("RGB", (384, 384), (255, 255, 255))

    # Process dataset
    def process_example(example):
        messages = json.loads(example["messages_json"]) if isinstance(example.get("messages_json"), str) else example.get("messages", [])

        # Extract text from messages
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(f"{role}: {item['text']}")
            elif isinstance(content, str):
                text_parts.append(f"{role}: {content}")

        full_text = "\n".join(text_parts)

        # Get or create image
        image = dummy_image

        image_ref = example.get("image_ref", "")
        if image_ref and os.path.exists(str(image_ref)):
            try:
                image = Image.open(str(image_ref)).convert("RGB")
                image = image.resize((384, 384))
            except Exception:
                pass

        # Process with SmolVLM processor
        try:
            inputs = processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            )

            # Flatten batch dim
            result = {k: v.squeeze(0) for k, v in inputs.items()}
            result["labels"] = result["input_ids"].clone()
            return result
        except Exception as e:
            # Return None values that will be filtered
            return {"input_ids": None}

    log.info("Processing dataset...")
    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        num_proc=1,  # Image processing not safe with multiproc
    )

    # Filter out failed examples
    processed = processed.filter(lambda x: x.get("input_ids") is not None)
    log.info(f"Processed: {len(processed)} valid examples")

    if len(processed) == 0:
        log.error("No valid examples after processing!")
        sys.exit(1)

    # Split train/eval
    split = processed.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    log.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Training arguments
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        max_steps=20000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
    )

    # Resume
    resume_ckpt = None
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"))
    if checkpoints:
        resume_ckpt = str(checkpoints[-1])
        log.info(f"Resuming from: {resume_ckpt}")

    log.info("Starting VLM training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    log.info("✓ VLM training complete.")


if __name__ == "__main__":
    train()
