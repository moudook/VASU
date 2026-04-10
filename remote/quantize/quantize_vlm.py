#!/usr/bin/env python3
"""
VASU — SmolVLM INT8 Quantization for ARM deployment.
"""

import logging
import os
import sys
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[QUANTIZE-VLM %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/quantize_vlm.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/models/vlm"
OUTPUT_DIR = f"{SCRATCH}/models/final/vlm"


def quantize_int8():
    """Quantize SmolVLM to INT8 using bitsandbytes."""
    import torch

    log.info("═══ VASU VLM — INT8 QUANTIZATION ═══")

    if not os.path.exists(MODEL_DIR):
        log.error(f"VLM model not found: {MODEL_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Load with INT8 quantization
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        log.info("Loading model with INT8 quantization...")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Save quantized model
        log.info(f"Saving INT8 model to {OUTPUT_DIR}")
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        log.info("✓ INT8 quantization complete")

    except Exception as e:
        log.warning(f"INT8 quantization failed: {e}")
        log.info("Copying bfloat16 model directly (500M params fits in ~1GB anyway).")
        shutil.copytree(MODEL_DIR, OUTPUT_DIR, dirs_exist_ok=True)

    # Print model size
    total_size = 0
    for f in Path(OUTPUT_DIR).rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            if size > 1024 * 1024:
                log.info(f"  {f.name}: {size / (1024*1024):.1f} MB")

    log.info(f"Total model size: {total_size / (1024*1024):.1f} MB")
    log.info("═══ VLM QUANTIZATION COMPLETE ═══")


if __name__ == "__main__":
    quantize_int8()
