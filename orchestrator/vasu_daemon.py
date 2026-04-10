#!/usr/bin/env python3
"""
VASU — Main Orchestrator Daemon
State machine that manages the full voice assistant pipeline on Redmi 7A.
Runs permanently as a systemd service.
"""

import enum
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Local module imports
from model_manager import ModelManager
from resource_manager import ResourceManager
from tool_handler import ToolHandler
from stt_client import STTClient
from tts_client import TTSClient
from wake_word import WakeWordListener

logging.basicConfig(
    level=logging.INFO,
    format="[VASU %(asctime)s] %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("/var/vasu/logs/vasu_daemon.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = "/etc/vasu/config.json"
MODELS_DIR = "/opt/vasu/models"


class VasuState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    LLM_THINKING = "llm_thinking"
    SWAPPING_VLM = "swapping_vlm"
    VLM_ACTIVE = "vlm_active"
    SWAPPING_LLM = "swapping_llm"
    SPEAKING = "speaking"
    ERROR = "error"


class VasuDaemon:
    """Main orchestrator state machine."""

    def __init__(self):
        self.state = VasuState.IDLE
        self.running = True
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 turns

        # Load config
        self.config = self._load_config()

        # Initialize subsystems
        self.resource_mgr = ResourceManager()
        self.model_mgr = ModelManager(MODELS_DIR)
        self.tool_handler = ToolHandler()
        self.stt = STTClient()
        self.tts = TTSClient()
        self.wake_word = WakeWordListener(callback=self._on_wake_word)

        # State lock for thread safety
        self._state_lock = threading.Lock()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _load_config(self) -> dict:
        """Load config from file or return defaults."""
        defaults = {
            "wake_word": "hey vasu",
            "language": "hi",
            "max_response_tokens": 256,
            "llm_model": "vasu_llm.gguf",
            "stt_model": "vasu_stt.onnx",
            "tts_model": "vasu_tts.onnx",
            "vlm_model_dir": "vasu_vlm",
            "silence_timeout_sec": 2.0,
            "thermal_throttle_temp": 70,
            "thermal_pause_temp": 80,
        }
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    user_config = json.load(f)
                defaults.update(user_config)
            except Exception as e:
                log.warning(f"Failed to load config: {e}")
        return defaults

    def _set_state(self, new_state: VasuState):
        """Thread-safe state transition."""
        with self._state_lock:
            old = self.state
            self.state = new_state
            log.info(f"State: {old.value} → {new_state.value}")

            # Adjust CPU governor based on state
            if new_state == VasuState.IDLE:
                self.resource_mgr.set_power_mode("powersave")
            else:
                self.resource_mgr.set_power_mode("performance")

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        log.info("Shutting down Vasu daemon...")
        self.running = False
        self.wake_word.stop()
        self.model_mgr.unload_all()
        self.resource_mgr.set_power_mode("powersave")
        log.info("Shutdown complete.")
        sys.exit(0)

    def _on_wake_word(self):
        """Called when wake word is detected."""
        if self.state != VasuState.IDLE:
            log.info("Wake word heard but not idle — ignoring")
            return

        log.info("Wake word detected!")
        self._set_state(VasuState.LISTENING)
        self._process_interaction()

    def _process_interaction(self):
        """Full interaction pipeline: listen → think → (optionally swap VLM) → speak."""
        try:
            # Step 1: Listen
            self._set_state(VasuState.LISTENING)
            audio_path = self.stt.record_until_silence(
                timeout_sec=self.config["silence_timeout_sec"]
            )

            if not audio_path:
                log.info("No speech detected, returning to idle")
                self._set_state(VasuState.IDLE)
                return

            # Step 2: Transcribe
            transcript = self.stt.transcribe(audio_path)
            if not transcript or len(transcript.strip()) < 2:
                log.info("Empty transcript, returning to idle")
                self._set_state(VasuState.IDLE)
                return

            log.info(f"User: {transcript}")

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": transcript})
            self._trim_history()

            # Step 3: Check thermal before heavy compute
            temp = self.resource_mgr.get_temperature()
            if temp > self.config["thermal_pause_temp"] * 1000:
                log.warning(f"CPU too hot ({temp/1000:.0f}°C), waiting for cooldown...")
                self.tts.speak("Ek second ruko, thoda garam ho gaya hai.")
                self.resource_mgr.wait_for_cooldown(self.config["thermal_throttle_temp"] * 1000)

            # Step 4: Ensure LLM is loaded
            if not self.model_mgr.is_llm_loaded():
                self.model_mgr.load_llm()

            # Step 5: Generate response
            self._set_state(VasuState.LLM_THINKING)
            response = self.model_mgr.generate_llm(
                self.conversation_history,
                max_tokens=self.config["max_response_tokens"],
            )

            if not response:
                log.error("LLM returned empty response")
                self.tts.speak("Sorry, kuch problem ho gaya. Phir se bolo.")
                self._set_state(VasuState.IDLE)
                return

            log.info(f"Vasu: {response}")

            # Step 6: Check for VLM invocation
            if "[INVOKE_VLM:" in response or "[invoke_camera" in response.lower():
                vlm_result = self._handle_vlm_invocation(response)
                if vlm_result:
                    # Re-generate with VLM context
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"[VASU_VISION: {vlm_result}]",
                    })
                    self._set_state(VasuState.LLM_THINKING)
                    response = self.model_mgr.generate_llm(
                        self.conversation_history,
                        max_tokens=self.config["max_response_tokens"],
                    )

            # Step 7: Check for tool calls
            tool_result = self.tool_handler.try_execute(response)
            if tool_result:
                self.conversation_history.append({
                    "role": "system",
                    "content": f"[TOOL_RESULT: {tool_result}]",
                })
                # Optionally re-generate to incorporate tool result
                if tool_result.get("needs_followup", False):
                    response = self.model_mgr.generate_llm(
                        self.conversation_history,
                        max_tokens=self.config["max_response_tokens"],
                    )

            # Step 8: Speak response
            self._set_state(VasuState.SPEAKING)
            # Clean response of any tool/VLM markers
            clean_response = self._clean_response(response)
            self.conversation_history.append({"role": "assistant", "content": clean_response})

            # Stream TTS sentence by sentence
            self.tts.speak(clean_response)

        except Exception as e:
            log.error(f"Interaction error: {e}", exc_info=True)
            self._set_state(VasuState.ERROR)
            try:
                self.tts.speak("Ek error aa gaya. Phir se try karo.")
            except Exception:
                pass

        finally:
            self._set_state(VasuState.IDLE)

    def _handle_vlm_invocation(self, response: str) -> str:
        """Handle LLM → VLM → LLM swap sequence."""
        import re

        log.info("VLM invocation detected")

        # Parse camera and query parameters
        camera = "rear"
        query = "describe what you see"

        # Extract from [INVOKE_VLM: query="...", camera="..."]
        match = re.search(r'query\s*=\s*"([^"]*)"', response)
        if match:
            query = match.group(1)
        match = re.search(r'camera\s*=\s*"([^"]*)"', response)
        if match:
            camera = match.group(1)

        # Also handle tool call JSON format
        try:
            tool_match = re.search(r'\{[^{}]*"invoke_camera"[^{}]*\}', response)
            if tool_match:
                tool_json = json.loads(tool_match.group())
                query = tool_json.get("params", {}).get("query", query)
                camera = tool_json.get("params", {}).get("camera", camera)
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            # Step 1: Save LLM context
            self._set_state(VasuState.SWAPPING_VLM)
            self.model_mgr.save_llm_context("/tmp/vasu_ctx.bin")
            self.model_mgr.unload_llm()

            # Step 2: Wait for RAM to be free
            self.resource_mgr.wait_for_ram(min_free_mb=1000, timeout_sec=3)

            # Step 3: Capture image
            image_path = "/tmp/capture.jpg"
            cam_source = "0" if camera == "rear" else "1"
            subprocess.run(
                ["libcamera-still", "-o", image_path, "--immediate",
                 "--timeout", "1000", "--camera", cam_source],
                timeout=5, capture_output=True,
            )

            if not os.path.exists(image_path):
                log.warning("Camera capture failed")
                return "Camera capture failed. Cannot see."

            # Step 4: Load VLM and run inference
            self._set_state(VasuState.VLM_ACTIVE)
            self.model_mgr.load_vlm()
            vlm_result = self.model_mgr.run_vlm(image_path, query)

            # Step 5: Save result, unload VLM
            Path("/tmp/vlm_result.txt").write_text(vlm_result)
            self.model_mgr.unload_vlm()

            # Step 6: Reload LLM + restore context
            self._set_state(VasuState.SWAPPING_LLM)
            self.resource_mgr.wait_for_ram(min_free_mb=800, timeout_sec=3)
            self.model_mgr.load_llm()
            self.model_mgr.restore_llm_context("/tmp/vasu_ctx.bin")

            log.info(f"VLM result: {vlm_result[:100]}...")
            return vlm_result

        except Exception as e:
            log.error(f"VLM swap failed: {e}")
            # Recover: make sure LLM is loaded
            try:
                self.model_mgr.load_llm()
            except Exception:
                pass
            return f"Vision error: {str(e)}"

    def _clean_response(self, response: str) -> str:
        """Remove tool call markers from response before speaking."""
        import re
        # Remove [INVOKE_VLM: ...] blocks
        response = re.sub(r'\[INVOKE_VLM:[^\]]*\]', '', response)
        # Remove JSON tool calls
        response = re.sub(r'\{"tool":[^}]*\}', '', response)
        # Remove [VASU_VISION: ...] blocks
        response = re.sub(r'\[VASU_VISION:[^\]]*\]', '', response)
        return response.strip()

    def _trim_history(self):
        """Keep conversation history within limits."""
        if len(self.conversation_history) > self.max_history * 2:
            # Keep system messages + last N turns
            system_msgs = [m for m in self.conversation_history if m["role"] == "system"]
            other_msgs = [m for m in self.conversation_history if m["role"] != "system"]
            self.conversation_history = system_msgs[-2:] + other_msgs[-self.max_history * 2:]

    def run(self):
        """Main daemon loop."""
        log.info("╔══════════════════════════════════════╗")
        log.info("║      VASU DAEMON STARTING            ║")
        log.info("╚══════════════════════════════════════╝")

        # Initialize resources
        self.resource_mgr.setup_cgroups()
        self.resource_mgr.set_power_mode("powersave")

        # Pre-load LLM on startup
        log.info("Pre-loading LLM...")
        try:
            self.model_mgr.load_llm()
            log.info("LLM loaded successfully")
        except Exception as e:
            log.error(f"Failed to pre-load LLM: {e}")

        # Start wake word listener
        log.info("Starting wake word listener...")
        self.wake_word.start()

        # Main loop — just keep alive and handle health checks
        while self.running:
            try:
                time.sleep(5)

                # Health check
                temp = self.resource_mgr.get_temperature()
                ram = self.resource_mgr.get_free_ram_mb()
                log.debug(f"Health: state={self.state.value}, temp={temp/1000:.0f}°C, "
                          f"free_ram={ram}MB")

            except KeyboardInterrupt:
                self._shutdown()
            except Exception as e:
                log.error(f"Main loop error: {e}")
                time.sleep(10)


def main():
    # Create required directories
    os.makedirs("/var/vasu/logs", exist_ok=True)
    os.makedirs("/var/vasu/notes", exist_ok=True)
    os.makedirs("/etc/vasu", exist_ok=True)

    daemon = VasuDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
