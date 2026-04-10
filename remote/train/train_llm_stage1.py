#!/usr/bin/env python3
"""
VASU — LLM Training Stage 1: Conversational SFT
Fine-tunes Qwen3-1.7B on conversational Hinglish data using QLoRA via Unsloth.
Distributed across 8x MI300X GPUs.
"""

import logging
import os
import sys
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-LLM-S1 %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_llm_stage1.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/llm/processed/stage1_sft"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/llm/stage1"
OUTPUT_DIR = f"{SCRATCH}/models/llm/stage1"

BASE_MODEL = "Qwen/Qwen3-1.7B"
MAX_SEQ_LENGTH = 4096


def train():
    import torch
    from datasets import load_from_disk

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU LLM — STAGE 1: CONVERSATIONAL SFT      ║")
    log.info("╚══════════════════════════════════════════════╝")

    num_gpus = torch.cuda.device_count()
    log.info(f"GPUs available: {num_gpus}")

    # Load dataset
    log.info(f"Loading dataset from {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        log.error(f"Dataset not found: {DATA_DIR}")
        sys.exit(1)
    dataset = load_from_disk(DATA_DIR)
    log.info(f"Dataset size: {len(dataset)}")

    # Try Unsloth first (2x faster, 70% less VRAM)
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        log.info("Using Unsloth for accelerated training")
        use_unsloth = True
    except ImportError:
        log.info("Unsloth not available. Using standard PEFT + transformers.")

    hf_token = os.environ.get("HF_TOKEN")

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            token=hf_token,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            max_seq_length=MAX_SEQ_LENGTH,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize dataset
    def tokenize_fn(examples):
        texts = []
        for messages in examples["messages"]:
            if isinstance(messages, str):
                messages = json.loads(messages)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    log.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    # Training arguments
    from transformers import TrainingArguments, Trainer

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        report_to="none",  # Set to "wandb" to enable W&B
        # run_name="vasu-llm-stage1",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if available
    resume_ckpt = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"))
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            log.info(f"Resuming from checkpoint: {resume_ckpt}")

    log.info("Starting Stage 1 training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    log.info(f"Saving model to {OUTPUT_DIR}")
    if use_unsloth:
        model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")
    else:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    log.info("✓ Stage 1 training complete.")


if __name__ == "__main__":
    train()
