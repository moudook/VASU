#!/usr/bin/env python3
"""
VASU - Model Manager
Handles loading, unloading, and hot-swapping of LLM and VLM models.
Uses llama.cpp for LLM and transformers for VLM.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

MODELS_DIR = "/opt/vasu/models"
CTX_SAVE_PATH = "/tmp/vasu_ctx.bin"

VASU_SYSTEM_PROMPT = (
    "You are Vasu, a home AI assistant. Rules:\n"
    "- Speak Hinglish naturally. Match user's language.\n"
    "- Keep responses SHORT: 1-3 sentences max.\n"
    "- Never use filler phrases.\n"
    "- Answer directly. No disclaimers.\n"
    '- For tools, output JSON: {"tool": "name", "params": {...}}\n'
    '- To see something, output: [INVOKE_VLM: query="what to look for", camera="rear"]'
)


class ModelManager:
    """Manages LLM/VLM loading, inference, and hot-swapping."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.llm = None
        self.llm_loaded = False
        self.vlm_model = None
        self.vlm_processor = None
        self.vlm_loaded = False

        self.llm_path = os.path.join(models_dir, "vasu_llm.gguf")
        self.vlm_dir = os.path.join(models_dir, "vasu_vlm")

    def _force_gc(self):
        """Force garbage collection to free RAM."""
        gc.collect()
        self._try_malloc_trim()

    @staticmethod
    def _try_malloc_trim():
        """Try to call malloc_trim on glibc or musl libc (multi-platform)."""
        import ctypes
        import ctypes.util

        libc_paths = [
            ctypes.util.find_library('c'),        # Most portable
            '/lib/libc.so.6',                     # glibc (standard Linux)
            '/lib/x86_64-linux-gnu/libc.so.6',   # Debian/Ubuntu
            '/usr/lib/libc.so',
            '/lib/libc.musl-x86_64.so.1',        # Alpine/postmarketOS x86
            '/usr/lib/libc.musl-x86_64.so.1',
            '/lib/libc.musl-aarch64.so.1',        # Alpine/postmarketOS ARM64
            '/usr/lib/libc.musl-aarch64.so.1',
            '/lib/libc.musl-armv7.so.1',          # Alpine/postmarketOS ARMv7 (Redmi 7A)
            '/usr/lib/libc.musl-armv7.so.1',
        ]
        for path in libc_paths:
            if path and os.path.exists(str(path)):
                try:
                    libc = ctypes.CDLL(str(path))
                    if hasattr(libc, 'malloc_trim'):
                        libc.malloc_trim(0)
                    return
                except OSError:
                    continue

    # ---- LLM (llama.cpp) ----

    def load_llm(self):
        """Load LLM via llama-cpp-python."""
        if self.llm_loaded:
            return

        log.info("Loading LLM: %s", self.llm_path)
        if not os.path.exists(self.llm_path):
            raise FileNotFoundError("LLM not found: " + self.llm_path)

        from llama_cpp import Llama

        self.llm = Llama(
            model_path=self.llm_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
            use_mmap=True,
        )
        self.llm_loaded = True
        log.info("LLM loaded successfully")

    def unload_llm(self):
        """Unload LLM from memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.llm_loaded = False
            self._force_gc()
            log.info("LLM unloaded")

    def is_llm_loaded(self):
        return self.llm_loaded

    def save_llm_context(self, path=CTX_SAVE_PATH):
        """Save LLM KV cache state."""
        if self.llm is None:
            return
        try:
            state = self.llm.save_state()
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"state": state, "ts": time.time()}, f)
            log.info("LLM context saved")
        except Exception as e:
            log.warning("Failed to save LLM context: %s", e)

    def restore_llm_context(self, path=CTX_SAVE_PATH):
        """Restore LLM KV cache."""
        if self.llm is None or not os.path.exists(path):
            return
        try:
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.llm.load_state(data["state"])
            log.info("LLM context restored")
        except Exception as e:
            log.warning("Failed to restore LLM context: %s", e)

    def _build_prompt(self, conversation):
        """Build prompt string from conversation history."""
        parts = ["System: " + VASU_SYSTEM_PROMPT + "\n"]
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append("User: " + content)
            elif role == "assistant":
                parts.append("Vasu: " + content)
            elif role == "system":
                parts.append("System: " + content)
        parts.append("Vasu:")
        return "\n".join(parts)

    def generate_llm(self, conversation, max_tokens=256):
        """Generate text from LLM given conversation history."""
        if not self.llm_loaded or self.llm is None:
            self.load_llm()

        prompt = self._build_prompt(conversation)

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "\nUser:"],
                echo=False,
            )
            text = output["choices"][0]["text"].strip()
            return text
        except Exception as e:
            log.error("LLM generation failed: %s", e)
            return ""

    # ---- VLM (SmolVLM via transformers) ----

    def load_vlm(self):
        """Load SmolVLM for vision inference."""
        if self.vlm_loaded:
            return

        log.info("Loading VLM: %s", self.vlm_dir)
        if not os.path.isdir(self.vlm_dir):
            raise FileNotFoundError("VLM not found: " + self.vlm_dir)

        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_dir, trust_remote_code=True,
        )
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            self.vlm_dir,
            torch_dtype=torch.float32,  # CPU inference, no bfloat16
            device_map="cpu",
            trust_remote_code=True,
        )
        self.vlm_model.eval()
        self.vlm_loaded = True
        log.info("VLM loaded successfully")

    def unload_vlm(self):
        """Unload VLM from memory."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        self.vlm_loaded = False
        self._force_gc()
        log.info("VLM unloaded")

    def run_vlm(self, image_path, query):
        """Run VLM inference on an image."""
        if not self.vlm_loaded:
            self.load_vlm()

        from PIL import Image
        import torch

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((384, 384))

            prompt = "User: " + query + "\nAssistant:"

            inputs = self.vlm_processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )

            result = self.vlm_processor.decode(
                outputs[0], skip_special_tokens=True,
            )

            # Extract assistant response
            if "Assistant:" in result:
                result = result.split("Assistant:")[-1].strip()

            return result

        except Exception as e:
            log.error("VLM inference failed: %s", e)
            return "Vision error: could not analyze image."

    def unload_all(self):
        """Unload all models."""
        self.unload_llm()
        self.unload_vlm()
