#!/usr/bin/env python3
"""
VASU — TTS Data Preprocessor
Prepares WAV files + text for Piper VITS training.
Converts everything to LJSpeech format: 22050 Hz mono WAV + metadata.csv
"""

import csv
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[PREPROCESS-TTS %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/preprocess_tts.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASETS_DIR = "/scratch/vasu/datasets/tts"
OUTPUT_DIR = "/scratch/vasu/datasets/tts/processed"
WAV_DIR = f"{OUTPUT_DIR}/wavs"
SAMPLE_RATE = 22050


def convert_audio(input_path: str, output_path: str):
    """Convert audio to 22050 Hz mono WAV using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        log.warning(f"Audio conversion failed for {input_path}: {e}")
        return False


def normalize_hindi_text(text: str) -> str:
    """Normalize Hindi/Hinglish text for TTS."""
    import re

    text = text.strip()

    # Expand common number patterns
    number_map_hi = {
        "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
        "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
    }

    # Expand simple standalone digits
    def expand_digit(match):
        return " ".join(number_map_hi.get(d, d) for d in match.group())

    text = re.sub(r'\b\d{1,2}\b', expand_digit, text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    return text.strip()


def process_indic_tts():
    """Process AI4Bharat IndicTTS Hindi dataset."""
    from datasets import load_from_disk

    path = f"{DATASETS_DIR}/indic_tts_hi"
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning("IndicTTS Hindi not available")
        return []

    log.info("Processing IndicTTS Hindi...")
    entries = []

    try:
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            splits = list(ds.keys())
            ds = ds[splits[0]]

        for idx, example in enumerate(ds):
            audio = example.get("audio")
            text = example.get("text", example.get("sentence", example.get("transcription", "")))

            if not text or not audio:
                continue

            text = normalize_hindi_text(text)
            if len(text) < 5:
                continue

            wav_name = f"indic_{idx:06d}.wav"
            wav_path = os.path.join(WAV_DIR, wav_name)

            # Save audio to WAV
            try:
                import soundfile as sf
                import numpy as np

                audio_array = audio.get("array", None)
                sr = audio.get("sampling_rate", 16000)

                if audio_array is not None:
                    # Resample if needed
                    if sr != SAMPLE_RATE:
                        import librosa
                        audio_array = librosa.resample(
                            np.array(audio_array, dtype=np.float32),
                            orig_sr=sr,
                            target_sr=SAMPLE_RATE,
                        )
                    sf.write(wav_path, audio_array, SAMPLE_RATE)
                    entries.append((wav_name, text))
            except Exception as e:
                log.warning(f"Failed to save audio {idx}: {e}")
                continue

        log.info(f"  IndicTTS: {len(entries)} entries")
    except Exception as e:
        log.error(f"Failed to process IndicTTS: {e}")

    return entries


def write_metadata(entries: list, output_path: str):
    """Write LJSpeech-style metadata.csv."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for wav_name, text in entries:
            # LJSpeech format: id|text|normalized_text
            utterance_id = wav_name.rsplit(".", 1)[0]
            writer.writerow([utterance_id, text, text])
    log.info(f"✓ Metadata written: {output_path} ({len(entries)} entries)")


def create_phoneme_config():
    """Create Piper phonemizer configuration."""
    config = {
        "audio": {
            "sample_rate": SAMPLE_RATE,
        },
        "espeak": {
            "voice": "hi",
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
        "phoneme_type": "espeak",
        "phoneme_map": {},
        "phoneme_id_map": {},
    }

    config_path = f"{OUTPUT_DIR}/config.json"
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"✓ Piper config written: {config_path}")


def run_phonemizer():
    """Run espeak-ng phonemizer on the metadata to generate phoneme alignments."""
    log.info("Running phonemizer...")
    try:
        # Test espeak-ng
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning("espeak-ng not available. Phonemization will happen during training.")
            return

        # Piper handles phonemization internally during training
        log.info("espeak-ng available. Phonemization will be done by Piper during training.")
    except FileNotFoundError:
        log.warning("espeak-ng not found. Install with: apt-get install espeak-ng")


def process_gtsinger():
    """Process GTSinger singing dataset - adds playful musical quality to TTS voice."""
    path = f"{DATASETS_DIR}/gtsinger_en"
    if not os.path.exists(path) or not os.path.exists(f"{path}/.download_complete"):
        log.warning("GTSinger dataset not available")
        return []

    log.info("Processing GTSinger a cappella dataset...")
    entries = []

    audio_idx = 0

    # Walk through all audio files
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                # Prioritize vibrato samples for natural playful quality
                if "Vibrato" in root or "Breathy" in root or "Control_Group" in root:
                    audio_path = os.path.join(root, file)
                    
                    # Use neutral text for singing samples - Piper will learn the tone
                    text = "Natural expressive voice with warm tone."

                    output_filename = f"gtsinger_{audio_idx:06d}.wav"
                    output_path = os.path.join(WAV_DIR, output_filename)
                    
                    if convert_audio(audio_path, output_path):
                        entries.append((output_filename, text))
                        audio_idx += 1

    log.info(f"Processed {len(entries)} GTSinger singing samples")
    return entries


def main():
    log.info("╔══════════════════════════════════════╗")
    log.info("║   VASU TTS DATA PREPROCESSING        ║")
    log.info("╚══════════════════════════════════════╝")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WAV_DIR, exist_ok=True)

    all_entries = []

    # Process IndicTTS
    entries = process_indic_tts()
    all_entries.extend(entries)
    
    # Process GTSinger a cappella dataset for playful voice quality
    gts_entries = process_gtsinger()
    all_entries.extend(gts_entries)

    if not all_entries:
        log.error("No TTS training data available!")
        log.info("TTS training will need data from other sources.")
        return

    # Write metadata
    write_metadata(all_entries, f"{OUTPUT_DIR}/metadata.csv")

    # Create config
    create_phoneme_config()

    # Test phonemizer
    run_phonemizer()

    log.info(f"TTS preprocessing complete. Total utterances: {len(all_entries)}")


if __name__ == "__main__":
    main()
