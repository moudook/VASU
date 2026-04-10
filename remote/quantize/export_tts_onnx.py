#!/usr/bin/env python3
"""
VASU — Export Piper TTS to ONNX for ARM deployment.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[EXPORT-TTS %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/export_tts_onnx.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/models/tts"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/tts"
OUTPUT_DIR = f"{SCRATCH}/models/final/tts"
PIPER_DIR = f"{SCRATCH}/piper"


def find_best_checkpoint():
    """Find the best TTS checkpoint or model."""
    # Check for Piper native checkpoint
    piper_ckpts = sorted(Path(CHECKPOINT_DIR).rglob("*.ckpt"))
    if piper_ckpts:
        return str(piper_ckpts[-1]), "piper"

    # Check for PyTorch model
    pt_model = f"{MODEL_DIR}/vits_model.pt"
    if os.path.exists(pt_model):
        return pt_model, "pytorch"

    return None, None


def export_piper_onnx(checkpoint_path: str):
    """Export Piper checkpoint to ONNX using Piper's export script."""
    log.info(f"Exporting Piper checkpoint: {checkpoint_path}")

    export_script = f"{PIPER_DIR}/src/python/piper_train/export_onnx.py"
    if not os.path.exists(export_script):
        log.warning("Piper export script not found. Trying manual export.")
        return False

    cmd = [
        "python3", export_script,
        checkpoint_path,
        OUTPUT_DIR,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info("✓ Piper ONNX export successful")
        return True
    else:
        log.warning(f"Piper export failed: {result.stderr}")
        return False


def export_pytorch_onnx(model_path: str):
    """Export PyTorch VITS model to ONNX."""
    import torch

    log.info(f"Exporting PyTorch model: {model_path}")

    # Load the model architecture (must match training)
    class VITSModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=192, nhead=2, batch_first=True),
                num_layers=6,
            )
            self.phone_embed = torch.nn.Embedding(256, 192)
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(192, 512, 16, 8),
                torch.nn.LeakyReLU(0.1),
                torch.nn.ConvTranspose1d(512, 256, 16, 8),
                torch.nn.LeakyReLU(0.1),
                torch.nn.ConvTranspose1d(256, 1, 4, 2),
                torch.nn.Tanh(),
            )

        def forward(self, phoneme_ids):
            x = self.phone_embed(phoneme_ids)
            x = self.encoder(x)
            x = x.transpose(1, 2)
            audio = self.decoder(x)
            return audio.squeeze(1)

    model = VITSModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Export to ONNX
    dummy_input = torch.randint(0, 256, (1, 50))  # Batch of phoneme IDs
    onnx_path = os.path.join(OUTPUT_DIR, "vasu_tts.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["phoneme_ids"],
        output_names=["audio"],
        dynamic_axes={
            "phoneme_ids": {0: "batch", 1: "sequence"},
            "audio": {0: "batch", 1: "samples"},
        },
        opset_version=14,
    )

    log.info(f"✓ ONNX exported: {onnx_path}")

    # Copy config
    config_src = f"{MODEL_DIR}/config.json"
    config_dst = f"{OUTPUT_DIR}/config.json"
    if os.path.exists(config_src):
        import shutil
        shutil.copy2(config_src, config_dst)

    return True


def create_piper_voice_config():
    """Create Piper voice JSON config for deployment."""
    voice_config = {
        "key": "hi_IN-vasu-medium",
        "name": "vasu",
        "language": {
            "code": "hi_IN",
            "family": "hi",
            "region": "IN",
            "name_native": "हिन्दी",
            "name_english": "Hindi",
            "country_english": "India",
        },
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "audio": {
            "sample_rate": 22050,
            "quality": "medium",
        },
        "espeak": {
            "voice": "hi",
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
    }

    config_path = os.path.join(OUTPUT_DIR, "hi_IN-vasu-medium.onnx.json")
    with open(config_path, "w") as f:
        json.dump(voice_config, f, indent=2)
    log.info(f"✓ Voice config: {config_path}")


def main():
    log.info("═══ VASU TTS — ONNX EXPORT ═══")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_path, model_type = find_best_checkpoint()

    if checkpoint_path is None:
        log.error("No TTS model found to export!")
        sys.exit(1)

    log.info(f"Model type: {model_type}, path: {checkpoint_path}")

    success = False
    if model_type == "piper":
        success = export_piper_onnx(checkpoint_path)

    if not success and model_type == "pytorch":
        success = export_pytorch_onnx(checkpoint_path)
    elif not success:
        # Try pytorch export as last resort
        pt_model = f"{MODEL_DIR}/vits_model.pt"
        if os.path.exists(pt_model):
            success = export_pytorch_onnx(pt_model)

    if success:
        create_piper_voice_config()

    # Print output files
    for f in Path(OUTPUT_DIR).rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            log.info(f"  {f.name}: {size_mb:.1f} MB")

    log.info("═══ TTS EXPORT COMPLETE ═══")


if __name__ == "__main__":
    main()
