#!/usr/bin/env python3
"""
VASU — STT Data Preprocessor
Processes audio datasets for Whisper fine-tuning.
Resamples to 16kHz, normalizes, prepares for Seq2SeqTrainer.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[PREPROCESS-STT %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/preprocess_stt.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASETS_DIR = "/scratch/vasu/datasets/stt"
OUTPUT_DIR = "/scratch/vasu/datasets/stt/processed"


def process_common_voice():
    """Process Common Voice Hindi."""
    from datasets import load_from_disk, Audio

    path = f"{DATASETS_DIR}/common_voice_hi"
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning("Common Voice Hindi not available")
        return None

    log.info("Processing Common Voice Hindi...")
    try:
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            # Use train split
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

        # Cast audio to 16kHz
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        # Keep only needed columns
        keep_cols = {"audio", "sentence", "path"}
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)

        # Rename for consistency
        if "sentence" in ds.column_names:
            ds = ds.rename_column("sentence", "text")

        log.info(f"  Common Voice: {len(ds)} samples")
        return ds
    except Exception as e:
        log.error(f"Failed to process Common Voice: {e}")
        return None


def process_fleurs():
    """Process FLEURS Hindi (evaluation set)."""
    from datasets import load_from_disk, Audio

    path = f"{DATASETS_DIR}/fleurs_hi"
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning("FLEURS Hindi not available")
        return None

    log.info("Processing FLEURS Hindi...")
    try:
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            if "test" in ds:
                ds = ds["test"]
            elif "validation" in ds:
                ds = ds["validation"]
            else:
                ds = ds[list(ds.keys())[0]]

        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        keep_cols = {"audio", "transcription", "path"}
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)

        if "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")

        log.info(f"  FLEURS: {len(ds)} samples")
        return ds
    except Exception as e:
        log.error(f"Failed to process FLEURS: {e}")
        return None


def process_kathbath():
    """Process Kathbath Hindi."""
    from datasets import load_from_disk, Audio

    path = f"{DATASETS_DIR}/kathbath"
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning("Kathbath not available")
        return None

    log.info("Processing Kathbath...")
    try:
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        # Normalize column names
        col_map = {}
        for col in ds.column_names:
            if col.lower() in ("transcript", "transcription", "sentence", "text_column"):
                col_map[col] = "text"
        if col_map:
            for old, new in col_map.items():
                ds = ds.rename_column(old, new)

        log.info(f"  Kathbath: {len(ds)} samples")
        return ds
    except Exception as e:
        log.error(f"Failed to process Kathbath: {e}")
        return None


def prepare_whisper_features(ds, processor_name="openai/whisper-small"):
    """Prepare features for Whisper training."""
    from transformers import WhisperProcessor

    log.info("Preparing Whisper features...")
    processor = WhisperProcessor.from_pretrained(processor_name)

    def prepare_example(batch):
        audio = batch["audio"]
        # Compute log-mel spectrogram
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # Tokenize text
        labels = processor.tokenizer(batch["text"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    ds = ds.map(
        prepare_example,
        remove_columns=ds.column_names,
        num_proc=4,
    )
    return ds


def main():
    log.info("╔══════════════════════════════════════╗")
    log.info("║   VASU STT DATA PREPROCESSING        ║")
    log.info("╚══════════════════════════════════════╝")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from datasets import concatenate_datasets

    # Process all STT datasets
    train_datasets = []
    eval_ds = None

    cv_ds = process_common_voice()
    if cv_ds is not None:
        train_datasets.append(cv_ds)

    kathbath_ds = process_kathbath()
    if kathbath_ds is not None:
        train_datasets.append(kathbath_ds)

    fleurs_ds = process_fleurs()
    if fleurs_ds is not None:
        eval_ds = fleurs_ds

    if not train_datasets:
        log.error("No STT training data available!")
        return

    # Concatenate training data
    log.info("Concatenating training datasets...")
    # Ensure consistent columns
    common_cols = set(train_datasets[0].column_names)
    for ds in train_datasets[1:]:
        common_cols &= set(ds.column_names)

    aligned = []
    for ds in train_datasets:
        drop = [c for c in ds.column_names if c not in common_cols]
        if drop:
            ds = ds.remove_columns(drop)
        aligned.append(ds)

    train_ds = concatenate_datasets(aligned)
    train_ds = train_ds.shuffle(seed=42)
    log.info(f"Total training samples: {len(train_ds)}")

    # Save raw (pre-feature) datasets
    train_ds.save_to_disk(f"{OUTPUT_DIR}/train_raw")
    log.info(f"✓ Saved raw train set → {OUTPUT_DIR}/train_raw")

    if eval_ds is not None:
        eval_ds.save_to_disk(f"{OUTPUT_DIR}/eval_raw")
        log.info(f"✓ Saved raw eval set → {OUTPUT_DIR}/eval_raw")

    log.info("STT preprocessing complete.")


if __name__ == "__main__":
    main()
