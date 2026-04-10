#!/usr/bin/env python3
"""
VASU - STT Client
Whisper inference client for speech-to-text.
Records audio, detects end of speech via VAD, transcribes with Whisper ONNX.
"""

import logging
import os
import subprocess
import tempfile
import time
import wave

log = logging.getLogger(__name__)

STT_MODEL_PATH = "/opt/vasu/models/vasu_stt.onnx"
SAMPLE_RATE = 16000
CHANNELS = 1


class STTClient:
    """Speech-to-text using Whisper (ONNX or whisper.cpp)."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._engine = None

    def _load_model(self):
        """Load STT model (try ONNX first, then whisper.cpp)."""
        if self._engine is not None:
            return

        # Try ONNX Runtime
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            from transformers import WhisperProcessor

            model_dir = os.path.dirname(STT_MODEL_PATH)
            if not os.path.isdir(model_dir):
                model_dir = "/opt/vasu/models/stt"

            self.processor = WhisperProcessor.from_pretrained(model_dir)
            self.model = ORTModelForSpeechSeq2Seq.from_pretrained(model_dir)
            self._engine = "onnx"
            log.info("STT loaded: ONNX Runtime")
            return
        except Exception as e:
            log.debug("ONNX STT failed: %s", e)

        # Try whisper.cpp
        whisper_cpp = "/opt/vasu/bin/whisper-cpp"
        if os.path.exists(whisper_cpp):
            self._engine = "whisper_cpp"
            log.info("STT: using whisper.cpp")
            return

        # Fallback: transformers Whisper
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            import torch

            model_dir = "/opt/vasu/models/stt"
            self.processor = WhisperProcessor.from_pretrained(model_dir)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.float32,
            )
            self.model.eval()
            self._engine = "transformers"
            log.info("STT loaded: transformers (slow)")
        except Exception as e:
            log.error("No STT engine available: %s", e)
            self._engine = "none"

    def record_until_silence(self, timeout_sec=2.0, max_duration=15.0):
        """Record audio until silence is detected (VAD)."""
        audio_path = "/tmp/vasu_recording.wav"

        try:
            # Use arecord with a fixed duration first, then apply VAD
            duration = min(max_duration, 10)
            cmd = [
                "arecord",
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", str(CHANNELS),
                "-d", str(int(duration)),
                audio_path,
            ]
            subprocess.run(cmd, timeout=duration + 2, capture_output=True)

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                # Trim silence from the end using simple energy-based VAD
                trimmed = self._trim_silence(audio_path, timeout_sec)
                return trimmed if trimmed else audio_path

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.error("Recording failed: %s", e)

        return None

    def _trim_silence(self, audio_path, silence_threshold_sec=2.0):
        """Trim trailing silence from audio file."""
        try:
            import numpy as np

            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

            if len(audio) == 0:
                return None

            # Find last point where energy exceeds threshold
            frame_size = int(SAMPLE_RATE * 0.1)  # 100ms frames
            energies = []
            for i in range(0, len(audio) - frame_size, frame_size):
                frame = audio[i:i + frame_size]
                energies.append(np.abs(frame).mean())

            if not energies:
                return audio_path

            threshold = max(energies) * 0.05  # 5% of max energy
            last_voice = 0
            for i, e in enumerate(energies):
                if e > threshold:
                    last_voice = i

            # Cut audio after last_voice + silence_threshold
            cut_sample = int((last_voice + silence_threshold_sec / 0.1) * frame_size)
            cut_sample = min(cut_sample, len(audio))

            if cut_sample < SAMPLE_RATE:  # Less than 1 second of speech
                return None

            trimmed_audio = audio[:cut_sample].astype(np.int16)

            trimmed_path = "/tmp/vasu_trimmed.wav"
            with wave.open(trimmed_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(trimmed_audio.tobytes())

            return trimmed_path

        except Exception as e:
            log.warning("Trim failed: %s", e)
            return audio_path

    def transcribe(self, audio_path):
        """Transcribe audio file to text."""
        self._load_model()

        if self._engine == "none":
            return ""

        if self._engine == "whisper_cpp":
            return self._transcribe_whisper_cpp(audio_path)

        return self._transcribe_python(audio_path)

    def _transcribe_python(self, audio_path):
        """Transcribe using Python Whisper (ONNX or transformers)."""
        try:
            import numpy as np
            import librosa

            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            input_features = self.processor.feature_extractor(
                audio, sampling_rate=SAMPLE_RATE,
            ).input_features

            if self._engine == "onnx":
                import numpy as np
                predicted_ids = self.model.generate(
                    input_features=np.array(input_features),
                )
            else:
                import torch
                input_features = torch.tensor(input_features)
                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features=input_features)

            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True,
            )[0]

            return transcription.strip()

        except Exception as e:
            log.error("Transcription failed: %s", e)
            return ""

    def _transcribe_whisper_cpp(self, audio_path):
        """Transcribe using whisper.cpp CLI."""
        model_path = "/opt/vasu/models/vasu_stt_ggml.bin"
        whisper_bin = "/opt/vasu/bin/whisper-cpp"

        try:
            result = subprocess.run(
                [whisper_bin, "-m", model_path, "-f", audio_path,
                 "--language", "hi", "--no-timestamps", "--threads", "4"],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout.strip()
        except Exception as e:
            log.error("whisper.cpp failed: %s", e)
            return ""
