#!/usr/bin/env python3
"""
VASU — HuggingFace Push Utility
Pushes checkpoints and final models to HuggingFace.
Called by cron every 5 hours OR manually with --final flag.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[HF-PUSH %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/push_hf.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

HF_REPO = "moudook/VASU_Versatile_AI_System_for_Home_Understanding"
SCRATCH = "/scratch/vasu"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints"
MODELS_DIR = f"{SCRATCH}/models"
PUSH_TRACKER = f"{SCRATCH}/.pushed_checkpoints.json"


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set. Cannot push.")
        return None
    return token


def load_push_tracker():
    """Track which checkpoints have already been pushed."""
    if os.path.exists(PUSH_TRACKER):
        with open(PUSH_TRACKER) as f:
            return json.load(f)
    return {"pushed": []}


def save_push_tracker(tracker):
    with open(PUSH_TRACKER, "w") as f:
        json.dump(tracker, f, indent=2)


def push_to_hf(local_path: str, subfolder: str, token: str, commit_message: str):
    """Push a directory or file to HuggingFace repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Ensure repo exists
    try:
        create_repo(HF_REPO, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        log.warning(f"Repo creation check: {e}")

    local_path = Path(local_path)

    try:
        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=subfolder,
                repo_id=HF_REPO,
                commit_message=commit_message,
                token=token,
            )
        elif local_path.is_file():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{subfolder}/{local_path.name}",
                repo_id=HF_REPO,
                commit_message=commit_message,
                token=token,
            )
        log.info(f"✓ Pushed: {subfolder}")
        return True
    except Exception as e:
        log.error(f"✗ Push failed for {subfolder}: {e}")
        return False


def push_checkpoints(token: str):
    """Push any new checkpoints that haven't been pushed yet."""
    tracker = load_push_tracker()
    pushed = set(tracker["pushed"])

    # Scan for checkpoints
    checkpoint_patterns = {
        "llm/stage1": f"{CHECKPOINT_DIR}/llm/stage1/checkpoint-*",
        "llm/stage2": f"{CHECKPOINT_DIR}/llm/stage2/checkpoint-*",
        "llm/stage3": f"{CHECKPOINT_DIR}/llm/stage3/checkpoint-*",
        "llm/stage4": f"{CHECKPOINT_DIR}/llm/stage4/checkpoint-*",
        "stt": f"{CHECKPOINT_DIR}/stt/checkpoint-*",
        "tts": f"{CHECKPOINT_DIR}/tts/checkpoint-*",
        "vlm": f"{CHECKPOINT_DIR}/vlm/checkpoint-*",
    }

    new_pushes = 0
    for model_key, pattern in checkpoint_patterns.items():
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            continue

        # Push only the latest checkpoint per model
        latest = dirs[-1]
        ckpt_id = f"{model_key}/{os.path.basename(latest)}"

        if ckpt_id in pushed:
            continue

        step = os.path.basename(latest).replace("checkpoint-", "")
        tag = f"vasu-{model_key.replace('/', '-')}-ckpt-{step}"
        success = push_to_hf(
            latest,
            f"checkpoints/{tag}",
            token,
            f"Checkpoint: {tag}",
        )

        if success:
            pushed.add(ckpt_id)
            new_pushes += 1

    tracker["pushed"] = list(pushed)
    save_push_tracker(tracker)
    log.info(f"Pushed {new_pushes} new checkpoints.")


def push_final_models(token: str):
    """Push all final quantized/exported models."""
    final_models = {
        # LLM GGUF
        "vasu-llm-final-q4": f"{MODELS_DIR}/final/vasu_llm_q4_k_m.gguf",
        "vasu-llm-final-q5": f"{MODELS_DIR}/final/vasu_llm_q5_k_m.gguf",
        # STT ONNX
        "vasu-stt-final-onnx": f"{MODELS_DIR}/final/stt",
        # TTS ONNX
        "vasu-tts-final-onnx": f"{MODELS_DIR}/final/tts",
        # VLM
        "vasu-vlm-final": f"{MODELS_DIR}/final/vlm",
    }

    for tag, path in final_models.items():
        if not os.path.exists(path):
            log.warning(f"Final model not found: {path} — skipping {tag}")
            continue

        success = push_to_hf(path, f"models/{tag}", token, f"Final model: {tag}")
        if not success:
            # Retry once after 30 minutes
            log.info(f"Will retry {tag} in 30 minutes...")
            time.sleep(1800)
            push_to_hf(path, f"models/{tag}", token, f"Final model (retry): {tag}")


def main():
    parser = argparse.ArgumentParser(description="VASU HuggingFace Push")
    parser.add_argument("--final", action="store_true",
                        help="Push final quantized models (end of training)")
    parser.add_argument("--checkpoints-only", action="store_true",
                        help="Push only new checkpoints (cron mode)")
    args = parser.parse_args()

    token = get_hf_token()
    if not token:
        sys.exit(1)

    if args.final:
        log.info("═══ FINAL MODEL PUSH ═══")
        push_checkpoints(token)
        push_final_models(token)
    else:
        log.info("═══ CHECKPOINT PUSH (cron) ═══")
        push_checkpoints(token)


if __name__ == "__main__":
    main()
