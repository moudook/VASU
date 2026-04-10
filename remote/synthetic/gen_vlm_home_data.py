#!/usr/bin/env python3
"""
VASU — Home Scene VLM Synthetic Data Generator
Uses DeepSeek-R1-Distill-Llama-70B to generate QA pairs about home scenes.
For image generation, uses existing VLM datasets + augmentation since text-to-image
would consume too much GPU budget. Generates text-only QA pairs that reference
home-scene categories, to be combined with real images from the Cauldron/LLaVA datasets.
"""

import json
import logging
import os
import sys
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format="[GEN-VLM %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/gen_vlm_data.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
MODEL_DIR = f"{SCRATCH}/teacher_model"
OUTPUT_FILE = f"{SCRATCH}/datasets/synthetic/vlm_home_data.json"
TARGET_COUNT = 2000
BATCH_SIZE = 10

TEACHER_GGUF_REPO = "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"
TEACHER_GGUF_FILE = "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

SYSTEM_PROMPT = """Generate {batch_size} visual question-answer pairs for training a home AI vision model.

Each pair should describe a realistic indoor scene and ask a question about it,
as if a user is asking their AI home assistant to analyze a camera image.

Format each as a JSON object:
{{
  "messages": [
    {{"role": "user", "content": [{{"type": "image"}}, {{"type": "text", "text": "user question about the image"}}]}},
    {{"role": "assistant", "content": [{{"type": "text", "text": "assistant answer describing what it sees"}}]}}
  ],
  "scene_description": "detailed description of the scene for context"
}}

Scene category focus: {category}

Rules:
- Questions should be in Hinglish, Hindi, or English
- Answers should be SHORT and practical (1-3 sentences)
- Include common Indian home elements (rangoli, traditional furniture, steel utensils, chapati maker, etc.)
- Handle low-quality images: mention if visibility is affected
- Types of questions: object identification, counting, spatial relationships, OCR, safety check
- Never refuse to describe what's in the image

Output a JSON array of {batch_size} entries."""

SCENE_CATEGORIES = [
    "Indian living room with sofa, TV, decorations, rangoli, family photos on walls",
    "Indian kitchen with gas stove, steel utensils, spice boxes, chapati maker, pressure cooker",
    "bedroom with bed, wardrobe, study desk, books, phone charging on nightstand",
    "bathroom with bucket, geyser, washing machine, toiletries on shelf",
    "balcony or terrace with plants, drying clothes, view of neighborhood",
    "doorway/entrance — checking if door is open/closed, shoes at entrance, parcels",
    "dining area with table, food items, water glasses, newspaper",
    "home office or study corner with laptop, books, stationery, cable management",
    "children's room with toys, school bags, colorful walls, small furniture",
    "pooja room or corner with idols, diyas, flowers, agarbatti holder",
    "product labels and packaging — reading Hindi/English text on boxes, medicines, food packets",
    "handwritten notes, shopping lists, school homework on paper",
    "low-light and blurry scenes — poorly lit rooms, motion blur, noisy images",
    "scenes with people — counting family members, checking if someone is home",
    "outdoor view through window — weather, vehicles, neighborhood activity",
    "safety checks — gas stove left on, iron plugged in, water tap running, door unlocked",
    "pet-related — cat on sofa, dog in corner, fish tank, bird cage",
    "festival decorations — Diwali lights, Holi colors, Christmas tree, Eid spread",
    "appliance status — is AC on, is fan running, TV showing something, WiFi router lights",
    "medicine box — reading medicine names, expiry dates, dosage instructions in Hindi",
]


def download_teacher_if_needed():
    """Download teacher model if not present."""
    gguf_path = os.path.join(MODEL_DIR, TEACHER_GGUF_FILE)
    if os.path.exists(gguf_path):
        return gguf_path

    os.makedirs(MODEL_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN")
    try:
        return hf_hub_download(
            repo_id=TEACHER_GGUF_REPO, filename=TEACHER_GGUF_FILE,
            local_dir=MODEL_DIR, token=hf_token,
        )
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def load_model():
    """Load teacher model."""
    model_path = download_teacher_if_needed()

    if model_path and os.path.exists(model_path):
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1,
                        n_batch=512, verbose=False)
            return llm, "llamacpp"
        except Exception as e:
            log.warning(f"llama.cpp failed: {e}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    return (model, tokenizer), "transformers"


def generate(llm, engine: str, prompt: str) -> str:
    """Generate text."""
    if engine == "llamacpp":
        output = llm(prompt, max_tokens=4000, temperature=0.8, top_p=0.9)
        return output["choices"][0]["text"]
    else:
        import torch
        model, tokenizer = llm
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=4000, temperature=0.8, top_p=0.9, do_sample=True,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text: str) -> list:
    """Extract JSON array from text."""
    text = text.strip()
    for strategy in [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find('['):t.rfind(']') + 1]),
    ]:
        try:
            data = strategy(text)
            if isinstance(data, list):
                valid = []
                for item in data:
                    messages = item.get("messages", [])
                    if messages and len(messages) >= 2:
                        valid.append(item)
                return valid
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return []


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU — VLM HOME SCENE DATA GENERATOR        ║")
    log.info("╚══════════════════════════════════════════════╝")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                all_data = json.load(f)
            log.info(f"Resuming from {len(all_data)}")
        except Exception:
            all_data = []

    if len(all_data) >= TARGET_COUNT:
        log.info("Target met.")
        return

    llm, engine = load_model()
    log.info(f"Engine: {engine}")

    topic_idx = 0
    while len(all_data) < TARGET_COUNT:
        category = SCENE_CATEGORIES[topic_idx % len(SCENE_CATEGORIES)]
        remaining = min(BATCH_SIZE, TARGET_COUNT - len(all_data))

        prompt = SYSTEM_PROMPT.format(batch_size=remaining, category=category)

        log.info(f"Generating: category='{category[:50]}...', progress={len(all_data)}/{TARGET_COUNT}")

        try:
            text = generate(llm, engine, prompt)
            valid = parse_json(text)

            if valid:
                all_data.extend(valid)
                log.info(f"  → {len(valid)} valid ({len(all_data)}/{TARGET_COUNT})")

            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"Error: {e}")

        topic_idx += 1
        time.sleep(0.5)

    log.info(f"✓ Generated {len(all_data)} VLM home scene QA pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
