#!/usr/bin/env python3
"""
VASU — Dataset Downloader
Downloads ALL datasets for all 4 models to /scratch/vasu/datasets/
Handles retries, partial downloads, and parallelism.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[DOWNLOAD %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/download_all.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASETS_DIR = "/scratch/vasu/datasets"


def download_hf_dataset(name: str, subset: str = None, split: str = None,
                        save_dir: str = None, max_retries: int = 3):
    """Download a HuggingFace dataset with retries."""
    from datasets import load_dataset

    if save_dir is None:
        safe_name = name.replace("/", "_")
        save_dir = os.path.join(DATASETS_DIR, safe_name)

    marker = os.path.join(save_dir, ".download_complete")
    if os.path.exists(marker):
        log.info(f"✓ Already downloaded: {name} ({subset or 'all'})")
        return True

    os.makedirs(save_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Downloading {name} (subset={subset}, split={split}) — attempt {attempt}")
            kwargs = {"trust_remote_code": True}
            if subset:
                kwargs["name"] = subset
            if split:
                kwargs["split"] = split

            ds = load_dataset(name, **kwargs)
            ds.save_to_disk(save_dir)

            # Write completion marker
            Path(marker).write_text(f"completed={time.time()}\n")
            log.info(f"✓ Downloaded: {name} → {save_dir}")
            return True

        except Exception as e:
            log.warning(f"Attempt {attempt} failed for {name}: {e}")
            if attempt < max_retries:
                time.sleep(30 * attempt)
            else:
                log.error(f"✗ FAILED after {max_retries} attempts: {name}")
                return False


# ─────────────────────────────────────────────────────
# DATASET REGISTRY
# ─────────────────────────────────────────────────────

LLM_DATASETS = [
    # Stage 1: Conversational SFT
    {"name": "ai4bharat/sangraha", "subset": "verified", "save_dir": f"{DATASETS_DIR}/llm/sangraha"},
    {"name": "CohereForAI/aya_dataset", "save_dir": f"{DATASETS_DIR}/llm/aya"},
    {"name": "HuggingFaceH4/ultrachat_200k", "save_dir": f"{DATASETS_DIR}/llm/ultrachat"},
    {"name": "jondurbin/airoboros-3.1", "save_dir": f"{DATASETS_DIR}/llm/airoboros"},
    {"name": "teknium/OpenHermes-2.5", "save_dir": f"{DATASETS_DIR}/llm/openhermes"},
    {"name": "WizardLMTeam/WizardLM_evol_instruct_70k", "save_dir": f"{DATASETS_DIR}/llm/wizardlm"},
    {"name": "garage-bAInd/Open-Platypus", "save_dir": f"{DATASETS_DIR}/llm/platypus"},
    {"name": "meta-math/MetaMathQA", "save_dir": f"{DATASETS_DIR}/llm/metamath"},

    # Stage 2: Anti-sycophancy DPO
    {"name": "jondurbin/truthy-dpo-v0.1", "save_dir": f"{DATASETS_DIR}/llm/truthy_dpo"},

    # Stage 3: Tool calling
    {"name": "NousResearch/hermes-function-calling-v1", "save_dir": f"{DATASETS_DIR}/llm/hermes_fc"},
    {"name": "glaiveai/glaive-function-calling-v2", "save_dir": f"{DATASETS_DIR}/llm/glaive_fc"},
    {"name": "Salesforce/xlam-function-calling-60k", "save_dir": f"{DATASETS_DIR}/llm/xlam_fc"},

    # Stage 4: Reasoning
    {"name": "open-r1/OpenR1-Math-220k", "save_dir": f"{DATASETS_DIR}/llm/openr1_math"},
    {"name": "simplescaling/s1K-1.1", "save_dir": f"{DATASETS_DIR}/llm/s1k"},
    {"name": "bespokelabs/Bespoke-Stratos-17k", "save_dir": f"{DATASETS_DIR}/llm/stratos"},
]

STT_DATASETS = [
    {"name": "mozilla-foundation/common_voice_13_0", "subset": "hi",
     "save_dir": f"{DATASETS_DIR}/stt/common_voice_hi"},
    {"name": "google/fleurs", "subset": "hi_in",
     "save_dir": f"{DATASETS_DIR}/stt/fleurs_hi"},
    {"name": "AI4Bharat/kathbath", "save_dir": f"{DATASETS_DIR}/stt/kathbath"},
]

TTS_DATASETS = [
    # Piper TTS data — these may need special handling
    {"name": "AI4Bharat/IndicTTS", "subset": "hi",
     "save_dir": f"{DATASETS_DIR}/tts/indic_tts_hi"},
    # Playful voice training: 80 hours professional a cappella singing (no instruments)
    # Adds natural musicality, vibrato, and playful tonal quality to TTS voice
    {"name": "GTSinger/GTSinger", "subset": "en",
     "save_dir": f"{DATASETS_DIR}/tts/gtsinger_en"},
]

VLM_DATASETS = [
    {"name": "HuggingFaceTB/the_cauldron", "subset": "coco_qa",
     "save_dir": f"{DATASETS_DIR}/vlm/cauldron_coco"},
    {"name": "HuggingFaceTB/the_cauldron", "subset": "textvqa",
     "save_dir": f"{DATASETS_DIR}/vlm/cauldron_textvqa"},
    {"name": "HuggingFaceTB/the_cauldron", "subset": "docvqa",
     "save_dir": f"{DATASETS_DIR}/vlm/cauldron_docvqa"},
    {"name": "HuggingFaceTB/the_cauldron", "subset": "ai2d",
     "save_dir": f"{DATASETS_DIR}/vlm/cauldron_ai2d"},
    {"name": "lmms-lab/LLaVA-Instruct-150K", "save_dir": f"{DATASETS_DIR}/vlm/llava_instruct"},
]


def download_category(name: str, datasets: list):
    """Download all datasets in a category."""
    log.info(f"═══ Downloading {name} datasets ({len(datasets)}) ═══")
    results = []
    for ds_info in datasets:
        success = download_hf_dataset(
            name=ds_info["name"],
            subset=ds_info.get("subset"),
            split=ds_info.get("split"),
            save_dir=ds_info.get("save_dir"),
        )
        results.append((ds_info["name"], success))
    return results


def main():
    log.info("╔══════════════════════════════════════╗")
    log.info("║   VASU DATASET DOWNLOAD START        ║")
    log.info("╚══════════════════════════════════════╝")

    os.makedirs(DATASETS_DIR, exist_ok=True)

    all_results = []

    # Download in parallel by category using threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_category, "LLM", LLM_DATASETS): "LLM",
            executor.submit(download_category, "STT", STT_DATASETS): "STT",
            executor.submit(download_category, "TTS", TTS_DATASETS): "TTS",
            executor.submit(download_category, "VLM", VLM_DATASETS): "VLM",
        }
        for future in as_completed(futures):
            category = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                log.error(f"Category {category} failed: {e}")

    # Summary
    log.info("═══ DOWNLOAD SUMMARY ═══")
    success_count = sum(1 for _, s in all_results if s)
    fail_count = sum(1 for _, s in all_results if not s)
    for name, success in all_results:
        status = "✓" if success else "✗"
        log.info(f"  {status} {name}")
    log.info(f"Total: {success_count} succeeded, {fail_count} failed out of {len(all_results)}")

    if fail_count > 0:
        log.warning("Some datasets failed to download. Training will proceed with available data.")

    log.info("Download phase complete.")


if __name__ == "__main__":
    main()
