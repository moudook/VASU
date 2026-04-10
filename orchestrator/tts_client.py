#!/usr/bin/env python3
"""
VASU - TTS Client
Piper TTS inference for speech synthesis.
Streams audio sentence-by-sentence via ALSA aplay.
"""

import logging
import os
import re
import subprocess
import tempfile

log = logging.getLogger(__name__)

TTS_MODEL_PATH = "/opt/vasu/models/vasu_tts.onnx"
TTS_CONFIG_PATH = "/opt/vasu/models/hi_IN-vasu-medium.onnx.json"
PIPER_BIN = "/opt/vasu/bin/piper"
SAMPLE_RATE = 22050


class TTSClient:
    """Text-to-speech using Piper ONNX or espeak-ng fallback."""

    def __init__(self):
        self._engine = None
        self._detect_engine()

    def _detect_engine(self):
        """Detect available TTS engine."""
        # Try Piper binary
        if os.path.exists(PIPER_BIN):
            self._engine = "piper"
            log.info("TTS engine: Piper")
            return

        # Try Piper Python
        try:
            import piper
            self._engine = "piper_python"
            log.info("TTS engine: Piper Python")
            return
        except ImportError:
            pass

        # Try ONNX Runtime with model directly
        if os.path.exists(TTS_MODEL_PATH):
            try:
                import onnxruntime
                self._engine = "onnx"
                log.info("TTS engine: ONNX direct")
                return
            except ImportError:
                pass

        # Fallback: espeak-ng
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
            self._engine = "espeak"
            log.info("TTS engine: espeak-ng (fallback)")
        except (FileNotFoundError, subprocess.CalledProcessError):
            self._engine = "none"
            log.warning("No TTS engine available")

    def speak(self, text):
        """Synthesize and play text. Streams sentence by sentence."""
        if not text or self._engine == "none":
            return

        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            try:
                if self._engine == "piper":
                    self._speak_piper(sentence)
                elif self._engine == "piper_python":
                    self._speak_piper_python(sentence)
                elif self._engine == "onnx":
                    self._speak_onnx(sentence)
                elif self._engine == "espeak":
                    self._speak_espeak(sentence)
            except Exception as e:
                log.error("TTS failed for sentence: %s", e)

    def _split_sentences(self, text):
        """Split text into sentences for streaming TTS."""
        # Split on sentence-ending punctuation (Hindi + English)
        sentences = re.split(r'(?<=[.!?\u0964\u0965])\s+', text)
        # Also split on very long segments
        result = []
        for s in sentences:
            if len(s) > 200:
                # Split on commas for very long sentences
                parts = re.split(r'(?<=[,\u002C])\s+', s)
                result.extend(parts)
            else:
                result.append(s)
        return result

    def _speak_piper(self, text):
        """Synthesize using Piper binary and play via aplay."""
        try:
            # Piper outputs raw audio to stdout, pipe to aplay
            piper_cmd = [
                PIPER_BIN,
                "--model", TTS_MODEL_PATH,
                "--config", TTS_CONFIG_PATH,
                "--output-raw",
            ]

            aplay_cmd = [
                "aplay",
                "-r", str(SAMPLE_RATE),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
                "-q",  # quiet, no status
            ]

            piper_proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            piper_proc.stdin.write(text.encode("utf-8"))
            piper_proc.stdin.close()
            aplay_proc.wait(timeout=15)
            piper_proc.wait(timeout=5)

        except Exception as e:
            log.error("Piper TTS error: %s", e)
            self._speak_espeak(text)  # Fallback

    def _speak_piper_python(self, text):
        """Synthesize using Piper Python library."""
        try:
            from piper import PiperVoice
            import wave

            voice = PiperVoice.load(TTS_MODEL_PATH, TTS_CONFIG_PATH)

            wav_path = "/tmp/vasu_tts_out.wav"
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                voice.synthesize(text, wav_file)

            subprocess.run(
                ["aplay", "-q", wav_path],
                timeout=15, capture_output=True,
            )

        except Exception as e:
            log.error("Piper Python error: %s", e)
            self._speak_espeak(text)

    def _speak_onnx(self, text):
        """Synthesize using ONNX model directly."""
        try:
            import onnxruntime as ort
            import numpy as np
            import wave

            session = ort.InferenceSession(TTS_MODEL_PATH)

            # Phonemize text
            phoneme_ids = [ord(c) % 256 for c in text]
            phoneme_ids = np.array([phoneme_ids], dtype=np.int64)

            # Run inference
            outputs = session.run(None, {"phoneme_ids": phoneme_ids})
            audio = outputs[0].squeeze()

            # Normalize to int16
            audio = (audio * 32767).astype(np.int16)

            # Save and play
            wav_path = "/tmp/vasu_tts_out.wav"
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            subprocess.run(
                ["aplay", "-q", wav_path],
                timeout=15, capture_output=True,
            )

        except Exception as e:
            log.error("ONNX TTS error: %s", e)
            self._speak_espeak(text)

    def _speak_espeak(self, text):
        """Fallback: use espeak-ng for synthesis."""
        try:
            # Detect if text is Hindi or English
            hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
            lang = "hi" if hindi_chars > len(text) * 0.2 else "en"

            subprocess.run(
                ["espeak-ng", "-v", lang, "-s", "150", text],
                timeout=15, capture_output=True,
            )
        except Exception as e:
            log.error("espeak-ng failed: %s", e)
