#!/usr/bin/env python3
"""
VASU — STT Training: Whisper Small Hindi-English Fine-tune
Fine-tunes openai/whisper-small on Hindi + Hinglish audio data.
Distributed across 4 GPUs (STT doesn't need all 8).
"""

import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-STT %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_stt.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/stt/processed"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/stt"
OUTPUT_DIR = f"{SCRATCH}/models/stt"

BASE_MODEL = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for Whisper training."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, Any]:
        import torch

        # Split inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred):
    """Compute WER for evaluation."""
    import evaluate
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    wer_metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def train():
    import torch
    from datasets import load_from_disk, Audio
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU STT — WHISPER SMALL HINDI FINE-TUNE     ║")
    log.info("╚══════════════════════════════════════════════╝")

    num_gpus = torch.cuda.device_count()
    log.info(f"GPUs: {num_gpus}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # Set generation config for Hindi
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # Load dataset
    train_path = f"{DATA_DIR}/train_raw"
    eval_path = f"{DATA_DIR}/eval_raw"

    if not os.path.exists(train_path):
        log.error(f"Training data not found: {train_path}")
        sys.exit(1)

    train_ds = load_from_disk(train_path)
    eval_ds = load_from_disk(eval_path) if os.path.exists(eval_path) else None

    log.info(f"Train: {len(train_ds)} samples")
    if eval_ds:
        log.info(f"Eval: {len(eval_ds)} samples")

    # Prepare features
    def prepare_dataset(batch):
        audio = batch["audio"]
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        labels = processor.tokenizer(batch["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    log.info("Preparing training features...")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        num_proc=4,
    )

    if eval_ds:
        log.info("Preparing eval features...")
        eval_ds = eval_ds.map(
            prepare_dataset,
            remove_columns=eval_ds.column_names,
            num_proc=4,
        )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        max_steps=25000,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=2000 if eval_ds else None,
        save_steps=2000,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="wer" if eval_ds else None,
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics if eval_ds else None,
    )

    # Resume from checkpoint
    resume_ckpt = None
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"))
    if checkpoints:
        resume_ckpt = str(checkpoints[-1])
        log.info(f"Resuming from: {resume_ckpt}")

    log.info("Starting Whisper training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    log.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    log.info("✓ STT training complete.")


if __name__ == "__main__":
    train()
