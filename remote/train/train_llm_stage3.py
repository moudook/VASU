#!/usr/bin/env python3
"""
VASU — LLM Training Stage 3: Reasoning Distillation
Fine-tunes on reasoning traces with budget-forced thinking.
Teaches Vasu WHEN to think and HOW MUCH.
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-LLM-S3 %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_llm_stage3.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/llm/processed/stage3_reasoning"
PREV_MODEL = f"{SCRATCH}/models/llm/stage2"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/llm/stage3"
OUTPUT_DIR = f"{SCRATCH}/models/llm/stage3"
MAX_SEQ_LENGTH = 4096


def train():
    import torch
    from datasets import load_from_disk

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU LLM — STAGE 3: REASONING DISTILLATION  ║")
    log.info("╚══════════════════════════════════════════════╝")

    base_model = PREV_MODEL if os.path.exists(PREV_MODEL) else f"{SCRATCH}/models/llm/stage1"
    if not os.path.exists(base_model):
        base_model = "Qwen/Qwen3-1.7B"
    log.info(f"Base model: {base_model}")

    if not os.path.exists(DATA_DIR):
        log.error(f"Dataset not found: {DATA_DIR}")
        sys.exit(1)

    dataset = load_from_disk(DATA_DIR)
    log.info(f"Dataset: {len(dataset)} examples")

    hf_token = os.environ.get("HF_TOKEN")

    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16, load_in_4bit=True, token=hf_token,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth",
            max_seq_length=MAX_SEQ_LENGTH,
        )
        use_unsloth = True
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto",
            torch_dtype=torch.bfloat16, token=hf_token, trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        texts = []
        for messages in examples["messages"]:
            if isinstance(messages, str):
                messages = json.loads(messages)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH,
                              padding="max_length", return_tensors="pt")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=100,
                            remove_columns=dataset.column_names, num_proc=4)

    from transformers import TrainingArguments, Trainer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, tokenizer=tokenizer)

    resume_ckpt = None
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"))
    if checkpoints:
        resume_ckpt = str(checkpoints[-1])
        log.info(f"Resuming from: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    if use_unsloth:
        model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")
    else:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    log.info("✓ Stage 3 training complete.")


if __name__ == "__main__":
    train()
