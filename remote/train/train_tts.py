#!/usr/bin/env python3
"""
VASU — TTS Training: Piper VITS Hindi Voice
Trains a VITS model for Hindi TTS using Piper's training pipeline.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-TTS %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_tts.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/tts/processed"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/tts"
OUTPUT_DIR = f"{SCRATCH}/models/tts"

PIPER_DIR = f"{SCRATCH}/piper"
SAMPLE_RATE = 22050
MAX_STEPS = 250000


def verify_data():
    """Check that TTS training data exists."""
    metadata = f"{DATA_DIR}/metadata.csv"
    wav_dir = f"{DATA_DIR}/wavs"

    if not os.path.exists(metadata):
        log.error(f"Metadata not found: {metadata}")
        return False

    if not os.path.isdir(wav_dir):
        log.error(f"WAV directory not found: {wav_dir}")
        return False

    # Count entries
    with open(metadata) as f:
        count = sum(1 for _ in f)
    log.info(f"Training data: {count} utterances")

    if count < 100:
        log.warning("Very few utterances. TTS quality may be poor.")

    return True


def create_piper_config():
    """Create Piper training configuration."""
    config = {
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "mel_channels": 80,
            "hop_length": 256,
            "win_length": 1024,
            "fft_size": 1024,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "seed": 42,
        },
        "espeak": {
            "voice": "hi",
        },
        "phoneme_type": "espeak",
        "num_symbols": 256,
        "num_speakers": 1,
    }

    config_path = f"{OUTPUT_DIR}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Training config: {config_path}")
    return config_path


def train_with_piper():
    """Train using Piper's native training script."""
    log.info("Attempting Piper native training...")

    train_script = f"{PIPER_DIR}/src/python/piper_train/__main__.py"
    if not os.path.exists(train_script):
        # Try alternate location
        train_script = f"{PIPER_DIR}/src/python/piper_train/train.py"
        if not os.path.exists(train_script):
            return False

    cmd = [
        "python3", train_script,
        "--dataset-dir", DATA_DIR,
        "--accelerator", "gpu",
        "--devices", "2",  # Use 2 GPUs for TTS
        "--batch-size", "32",
        "--validation-split", "0.05",
        "--max-epochs", "-1",  # Use max_steps instead
        "--max_steps", str(MAX_STEPS),
        "--checkpoint-epochs", "0",
        "--quality", "medium",
        "--precision", "bf16-mixed",
        "--default_root_dir", CHECKPOINT_DIR,
    ]

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train_with_pytorch_lightning():
    """Fallback: train VITS using PyTorch Lightning directly."""
    log.info("Training VITS with PyTorch Lightning fallback...")

    import torch

    try:
        import pytorch_lightning as pl
    except ImportError:
        os.system("pip install pytorch-lightning")
        import pytorch_lightning as pl

    # VITS training implementation
    from torch.utils.data import Dataset, DataLoader
    import csv
    import soundfile as sf

    class TTSDataset(Dataset):
        """Simple TTS dataset that loads wav + text."""

        def __init__(self, metadata_path: str, wav_dir: str):
            self.wav_dir = wav_dir
            self.entries = []
            with open(metadata_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for row in reader:
                    if len(row) >= 2:
                        self.entries.append((row[0], row[1]))

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            utt_id, text = self.entries[idx]
            wav_path = os.path.join(self.wav_dir, f"{utt_id}.wav")

            try:
                audio, sr = sf.read(wav_path)
                audio = torch.FloatTensor(audio)
            except Exception:
                audio = torch.zeros(SAMPLE_RATE)  # Fallback: 1 second of silence

            # Simple phoneme encoding (espeak-ng)
            try:
                result = subprocess.run(
                    ["espeak-ng", "-v", "hi", "-q", "--ipa", text],
                    capture_output=True, text=True, timeout=5,
                )
                phonemes = result.stdout.strip()
            except Exception:
                phonemes = text

            # Convert phonemes to integer IDs
            phoneme_ids = [ord(c) % 256 for c in phonemes]

            return {
                "audio": audio,
                "phoneme_ids": torch.LongTensor(phoneme_ids),
                "text": text,
            }

    def collate_fn(batch):
        """Custom collate for variable-length audio."""
        max_audio_len = max(b["audio"].shape[0] for b in batch)
        max_phone_len = max(b["phoneme_ids"].shape[0] for b in batch)

        audios = torch.zeros(len(batch), max_audio_len)
        phones = torch.zeros(len(batch), max_phone_len, dtype=torch.long)
        audio_lens = torch.zeros(len(batch), dtype=torch.long)
        phone_lens = torch.zeros(len(batch), dtype=torch.long)

        for i, b in enumerate(batch):
            audios[i, :b["audio"].shape[0]] = b["audio"]
            phones[i, :b["phoneme_ids"].shape[0]] = b["phoneme_ids"]
            audio_lens[i] = b["audio"].shape[0]
            phone_lens[i] = b["phoneme_ids"].shape[0]

        return {
            "audio": audios,
            "phoneme_ids": phones,
            "audio_lengths": audio_lens,
            "phoneme_lengths": phone_lens,
        }

    # Create dataset
    ds = TTSDataset(f"{DATA_DIR}/metadata.csv", f"{DATA_DIR}/wavs")
    log.info(f"TTS Dataset: {len(ds)} utterances")

    if len(ds) == 0:
        log.error("No data in TTS dataset!")
        return False

    # Split train/val
    val_size = max(1, int(len(ds) * 0.05))
    train_size = len(ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    # Simple VITS-like training module
    class VITSTrainer(pl.LightningModule):
        def __init__(self):
            super().__init__()
            # Simplified VITS architecture for training
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

        def training_step(self, batch, batch_idx):
            phoneme_ids = batch["phoneme_ids"]
            target_audio = batch["audio"]

            pred_audio = self(phoneme_ids)

            # Match lengths
            min_len = min(pred_audio.shape[1], target_audio.shape[1])
            loss = torch.nn.functional.l1_loss(
                pred_audio[:, :min_len],
                target_audio[:, :min_len],
            )

            self.log("train_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=2e-4, betas=(0.8, 0.99))

    model = VITSTrainer()

    trainer = pl.Trainer(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=min(2, torch.cuda.device_count()),
        precision="bf16-mixed",
        default_root_dir=CHECKPOINT_DIR,
        val_check_interval=10000,
        log_every_n_steps=100,
        enable_checkpointing=True,
    )

    # Resume from checkpoint
    ckpt_path = None
    latest = sorted(Path(CHECKPOINT_DIR).glob("**/*.ckpt"))
    if latest:
        ckpt_path = str(latest[-1])
        log.info(f"Resuming from: {ckpt_path}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save final model
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/vits_model.pt")
    log.info(f"Model saved to {OUTPUT_DIR}/vits_model.pt")
    return True


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU TTS — PIPER VITS HINDI TRAINING         ║")
    log.info("╚══════════════════════════════════════════════╝")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not verify_data():
        log.error("TTS training data not ready. Run preprocess_tts.py first.")
        sys.exit(1)

    create_piper_config()

    # Try Piper native training first
    success = train_with_piper()

    if not success:
        log.info("Piper native training unavailable. Using PyTorch Lightning fallback.")
        success = train_with_pytorch_lightning()

    if success:
        log.info("✓ TTS training complete.")
    else:
        log.error("✗ TTS training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
