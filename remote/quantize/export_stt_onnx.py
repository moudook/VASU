#!/usr/bin/env python3
"""
VASU — Export Whisper STT to ONNX INT8 for ARM deployment.
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[EXPORT-STT %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/export_stt_onnx.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/models/stt"
OUTPUT_DIR = f"{SCRATCH}/models/final/stt"


def export_onnx():
    import torch

    log.info("═══ VASU STT — ONNX EXPORT ═══")

    if not os.path.exists(MODEL_DIR):
        log.error(f"STT model not found: {MODEL_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try optimum export first
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        log.info("Exporting via Optimum ONNX Runtime...")

        processor = WhisperProcessor.from_pretrained(MODEL_DIR)

        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            MODEL_DIR,
            export=True,
        )
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        log.info(f"✓ ONNX model saved: {OUTPUT_DIR}")

        # INT8 quantization
        log.info("Applying INT8 quantization...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            qconfig = AutoQuantizationConfig.avx2(is_static=False)

            encoder_path = os.path.join(OUTPUT_DIR, "encoder_model.onnx")
            decoder_path = os.path.join(OUTPUT_DIR, "decoder_model.onnx")

            for onnx_path in [encoder_path, decoder_path]:
                if os.path.exists(onnx_path):
                    quantizer = ORTQuantizer.from_pretrained(OUTPUT_DIR)
                    quantizer.quantize(save_dir=OUTPUT_DIR, quantization_config=qconfig)
                    log.info(f"  ✓ Quantized: {os.path.basename(onnx_path)}")

        except Exception as e:
            log.warning(f"INT8 quantization failed: {e}. Using FP32 ONNX.")

        return True

    except Exception as e:
        log.warning(f"Optimum export failed: {e}")

    # Fallback: manual ONNX export
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import onnx

        log.info("Manual ONNX export...")

        processor = WhisperProcessor.from_pretrained(MODEL_DIR)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32,
        )
        model.eval()

        # Export encoder
        dummy_input = torch.randn(1, 80, 3000)  # log-mel spectrogram
        encoder_path = os.path.join(OUTPUT_DIR, "encoder_model.onnx")

        torch.onnx.export(
            model.model.encoder,
            dummy_input,
            encoder_path,
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_features": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
            opset_version=14,
        )
        log.info(f"✓ Encoder exported: {encoder_path}")

        # Save processor
        processor.save_pretrained(OUTPUT_DIR)

        return True

    except Exception as e:
        log.error(f"Manual ONNX export also failed: {e}")

    # Last resort: copy the HF model as-is
    log.warning("Copying HF model directly (no ONNX conversion).")
    import shutil
    shutil.copytree(MODEL_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    return True


def main():
    export_onnx()
    log.info("═══ STT EXPORT COMPLETE ═══")

    # Print file sizes
    for f in Path(OUTPUT_DIR).rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            log.info(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
