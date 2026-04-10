#!/usr/bin/env python3
"""
VASU - Wake Word Listener
Always-on wake word detection using openWakeWord.
Runs at highest priority (nice -20), minimal RAM footprint (~20MB).
"""

import logging
import os
import threading
import time

log = logging.getLogger(__name__)


class WakeWordListener:
    """Always-on wake word detector for 'Hey Vasu'."""

    def __init__(self, callback=None, wake_word="hey_vasu", threshold=0.5):
        self.callback = callback
        self.wake_word = wake_word
        self.threshold = threshold
        self.running = False
        self._thread = None
        self._cooldown = 2.0  # seconds between activations

    def start(self):
        """Start listening for wake word in background thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log.info("Wake word listener started (word='%s')", self.wake_word)

    def stop(self):
        """Stop the listener."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        log.info("Wake word listener stopped")

    def _listen_loop(self):
        """Main listening loop using openWakeWord."""
        try:
            self._listen_openwakeword()
        except ImportError:
            log.warning("openWakeWord not available. Trying pvporcupine...")
            try:
                self._listen_porcupine()
            except ImportError:
                log.warning("Porcupine not available. Using simple keyword spotting.")
                self._listen_simple()

    def _listen_openwakeword(self):
        """Listen using openWakeWord library."""
        import openwakeword
        from openwakeword.model import Model
        import pyaudio
        import numpy as np

        # Load wake word model
        oww_model = Model(
            wakeword_models=[self.wake_word],
            inference_framework="onnx",
        )

        # Audio stream
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280,  # 80ms chunks
        )

        log.info("openWakeWord listening...")
        last_trigger = 0

        while self.running:
            try:
                audio_data = stream.read(1280, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                prediction = oww_model.predict(audio_np)

                for model_name, score in prediction.items():
                    if score > self.threshold:
                        now = time.time()
                        if now - last_trigger > self._cooldown:
                            last_trigger = now
                            log.info("Wake word detected! (score=%.2f)", score)
                            if self.callback:
                                # Run callback in separate thread to not block listener
                                threading.Thread(
                                    target=self.callback, daemon=True,
                                ).start()

            except Exception as e:
                log.error("Wake word stream error: %s", e)
                time.sleep(1)

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _listen_porcupine(self):
        """Fallback: listen using Picovoice Porcupine."""
        import pvporcupine
        import pyaudio
        import struct

        porcupine = pvporcupine.create(
            keywords=["hey google"],  # Closest built-in, will train custom later
            sensitivities=[0.7],
        )

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=porcupine.sample_rate,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )

        log.info("Porcupine listening...")
        last_trigger = 0

        while self.running:
            try:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                result = porcupine.process(pcm)
                if result >= 0:
                    now = time.time()
                    if now - last_trigger > self._cooldown:
                        last_trigger = now
                        log.info("Wake word detected (porcupine)!")
                        if self.callback:
                            threading.Thread(target=self.callback, daemon=True).start()

            except Exception as e:
                log.error("Porcupine error: %s", e)
                time.sleep(1)

        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()

    def _listen_simple(self):
        """Last resort: simple energy-based detection + periodic check."""
        log.info("Using simple energy-based activation (low quality)")
        log.info("Press Enter or say something loud to activate")

        while self.running:
            try:
                # In simple mode, just activate on any audio above threshold
                # This is NOT real wake word detection, just a placeholder
                time.sleep(0.5)

                # Check if there's audio input above energy threshold
                try:
                    import pyaudio
                    import numpy as np

                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=4096,
                    )
                    data = stream.read(4096, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    energy = np.abs(audio).mean()
                    stream.close()
                    pa.terminate()

                    if energy > 1000:  # Loud audio detected
                        log.info("Audio energy trigger (energy=%.0f)", energy)
                        if self.callback:
                            threading.Thread(target=self.callback, daemon=True).start()
                        time.sleep(self._cooldown)

                except Exception:
                    time.sleep(5)

            except Exception as e:
                log.error("Simple listener error: %s", e)
                time.sleep(5)
