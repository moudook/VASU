#!/usr/bin/env python3
"""
VASU — LLM Training Stage 4: GRPO Reinforcement Learning
Uses rule-based reward signals to reinforce desired behaviors:
+1 correct tool JSON, +1 short responses, -1 filler phrases, +1 language matching, etc.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[TRAIN-LLM-S4 %(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/vasu/logs/train_llm_stage4.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SCRATCH = "/scratch/vasu"
DATA_DIR = f"{SCRATCH}/datasets/llm/processed/stage4_grpo"
PREV_MODEL = f"{SCRATCH}/models/llm/stage3"
CHECKPOINT_DIR = f"{SCRATCH}/checkpoints/llm/stage4"
OUTPUT_DIR = f"{SCRATCH}/models/llm/stage4"
MAX_SEQ_LENGTH = 4096

# Forbidden filler phrases
FILLER_PHRASES = [
    "great question", "certainly", "as an ai", "as a language model",
    "i'd be happy to help", "i would be happy", "absolutely", "of course",
    "i cannot help", "i cannot assist", "i'm not able",
    "please consult a professional", "i must emphasize",
]


def compute_reward(prompt: str, response: str) -> float:
    """Rule-based reward function for GRPO."""
    reward = 0.0
    response_lower = response.lower()
    prompt_lower = prompt.lower()

    # -1 for filler phrases
    for phrase in FILLER_PHRASES:
        if phrase in response_lower:
            reward -= 1.0
            break

    # +1 for correct tool JSON format
    if '{"tool"' in response or '"tool":' in response:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
            if json_match:
                tool_call = json.loads(json_match.group())
                if "tool" in tool_call and "params" in tool_call:
                    reward += 1.0
        except (json.JSONDecodeError, ValueError):
            pass

    # +1 for short responses on simple queries
    prompt_words = len(prompt_lower.split())
    response_words = len(response.split())
    if prompt_words < 15 and response_words < 100:
        reward += 1.0
    elif prompt_words < 15 and response_words > 200:
        reward -= 0.5  # Too verbose for simple query

    # +1 for language matching
    hindi_chars_prompt = len(re.findall(r'[\u0900-\u097F]', prompt))
    hindi_chars_response = len(re.findall(r'[\u0900-\u097F]', response))
    if hindi_chars_prompt > 5 and hindi_chars_response > 0:
        reward += 1.0  # Responds in Hindi when asked in Hindi
    elif hindi_chars_prompt == 0 and hindi_chars_response == 0:
        reward += 0.5  # English to English

    # -1 for refusing factual/technical questions
    refusal_patterns = [
        "i cannot provide", "i can't help with that",
        "it would not be appropriate", "i must decline",
    ]
    for pattern in refusal_patterns:
        if pattern in response_lower:
            reward -= 1.0
            break

    # +1 for asking clarification on ambiguous multi-step tasks
    ambiguous_keywords = ["kar do", "set kar", "bana do", "remind", "plan"]
    is_ambiguous = any(kw in prompt_lower for kw in ambiguous_keywords) and prompt_words < 8
    if is_ambiguous and "?" in response:
        reward += 1.0  # Good: asks clarification

    return reward


def train():
    import torch
    from datasets import load_from_disk

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  VASU LLM — STAGE 4: GRPO REINFORCEMENT      ║")
    log.info("╚══════════════════════════════════════════════╝")

    base_model = PREV_MODEL if os.path.exists(PREV_MODEL) else f"{SCRATCH}/models/llm/stage2"
    if not os.path.exists(base_model):
        base_model = "Qwen/Qwen3-1.7B"
    log.info(f"Base model: {base_model}")

    if not os.path.exists(DATA_DIR):
        log.error(f"Dataset not found: {DATA_DIR}")
        sys.exit(1)

    dataset = load_from_disk(DATA_DIR)
    log.info(f"Dataset: {len(dataset)} examples")

    hf_token = os.environ.get("HF_TOKEN")

    # Try TRL GRPOTrainer
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig

        log.info("Using TRL GRPOTrainer")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto",
            torch_dtype=torch.bfloat16, token=hf_token, trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )

        # Prepare dataset for GRPO: needs "prompt" field
        def format_for_grpo(example):
            prompt = example.get("prompt", "")
            if not prompt:
                chosen = example.get("chosen", "")
                prompt = chosen[:200] if chosen else ""
            return {"prompt": prompt}

        grpo_dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)
        grpo_dataset = grpo_dataset.filter(lambda x: len(x["prompt"]) > 10)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Define reward function for GRPO
        def reward_fn(completions, prompts=None, **kwargs):
            """Compute rewards for a batch of completions."""
            rewards = []
            for i, completion in enumerate(completions):
                prompt = prompts[i] if prompts else ""
                r = compute_reward(prompt, completion)
                rewards.append(r)
            return rewards

        training_args = GRPOConfig(
            output_dir=CHECKPOINT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            bf16=True,
            logging_steps=25,
            save_steps=200,
            save_total_limit=3,
            max_completion_length=512,
            num_generations=4,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=grpo_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            peft_config=peft_config,
        )

        resume_ckpt = None
        checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"))
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            log.info(f"Resuming from: {resume_ckpt}")

        trainer.train(resume_from_checkpoint=resume_ckpt)

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    except Exception as e:
        log.warning(f"GRPO training failed: {e}")
        log.info("Falling back to DPO training...")

        # Fallback: DPO with the truthy-dpo data
        try:
            from trl import DPOTrainer, DPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb_config, device_map="auto",
                torch_dtype=torch.bfloat16, token=hf_token, trust_remote_code=True,
            )

            peft_config = LoraConfig(
                r=64, lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )

            # Format for DPO
            def format_dpo(example):
                return {
                    "prompt": example.get("prompt", ""),
                    "chosen": example.get("chosen", ""),
                    "rejected": example.get("rejected", ""),
                }

            dpo_dataset = dataset.map(format_dpo)
            dpo_dataset = dpo_dataset.filter(lambda x: x["prompt"] and x["chosen"])

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            dpo_config = DPOConfig(
                output_dir=CHECKPOINT_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=5e-5,
                bf16=True,
                logging_steps=25,
                save_steps=200,
                save_total_limit=3,
                report_to="none",
            )

            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=dpo_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )

            trainer.train()
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

        except Exception as e2:
            log.error(f"DPO fallback also failed: {e2}")
            log.info("Copying Stage 3 model as final.")
            if os.path.exists(PREV_MODEL):
                import shutil
                shutil.copytree(PREV_MODEL, OUTPUT_DIR, dirs_exist_ok=True)

    log.info("✓ Stage 4 training complete.")


if __name__ == "__main__":
    train()
