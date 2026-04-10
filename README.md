# VASU — Versatile AI System for Home Understanding

> A complete, self-contained AI home assistant that runs entirely on a Xiaomi Redmi 7A (2GB RAM, Snapdragon 439) with voice-only interface. No cloud. No internet required for core features.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VOICE PIPELINE                          │
│                                                             │
│  Mic → [Wake Word] → [STT/Whisper] → [LLM/Qwen3] → [TTS]  │
│              20MB         150MB          950MB         80MB  │
│                                                             │
│         ┌──── Hot-swap slot (1.35GB) ────┐                  │
│         │  LLM (default) ←→ VLM (camera) │                  │
│         └────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Four Models

| Model | Base | Size (RAM) | Target |
|-------|------|-----------|--------|
| VASU-LLM | Qwen3-1.7B Q4_K_M | ~950MB | Hinglish conversation, tools, reasoning |
| VASU-STT | Whisper Small | ~150MB | Hindi-English speech recognition |
| VASU-TTS | Piper VITS | ~80MB | Hindi voice synthesis, <100ms latency |
| VASU-VLM | SmolVLM-500M | ~900MB | Home scene understanding, OCR |

## Quick Start

### 1. Deploy to GPU Cluster

```bash
export HF_TOKEN="hf_your_write_token"
python deploy.py --ip <GPU_DROPLET_IP> --key ~/.ssh/id_rsa
```

This will:
- Upload all training code to the droplet
- Install ROCm + PyTorch + all dependencies
- Download datasets, generate synthetic data
- Train all 4 models sequentially
- Push checkpoints to HuggingFace every 5 hours
- Push final quantized models when done

### 2. Deploy to Phone

After training completes:

```bash
# Download models from HuggingFace to USB drive
# Flash postmarketOS on Redmi 7A
# Run device setup:
bash device/setup_device.sh

# Start Vasu:
systemctl start vasu-orchestrator
```

## Project Structure

```
vasu_project/
├── deploy.py                    # Entry point — SSH deploy to GPU cluster
├── remote/                      # Runs on GPU droplet
│   ├── setup.sh                 # Install all dependencies
│   ├── master_run.sh            # Orchestrates full pipeline
│   ├── push_hf.py               # HuggingFace push utility
│   ├── cron_setup.sh            # 5-hour push cron job
│   ├── data/                    # Download + preprocess
│   ├── train/                   # 7 training scripts (4 LLM stages + STT + TTS + VLM)
│   ├── quantize/                # GGUF, ONNX export
│   └── synthetic/               # Synthetic data generation (DeepSeek-R1 teacher)
├── orchestrator/                # On-device Python daemon
│   ├── vasu_daemon.py           # Main state machine
│   ├── model_manager.py         # LLM/VLM hot-swap
│   ├── resource_manager.py      # CPU, thermal, RAM management
│   ├── tool_handler.py          # Tool execution (camera, alarm, notes, etc.)
│   ├── wake_word.py             # Always-on wake word listener
│   ├── stt_client.py            # Whisper inference
│   └── tts_client.py            # Piper inference
└── device/                      # Phone setup
    ├── setup_device.sh          # Full device setup
    ├── systemd/                 # 4 service files
    └── libcamera_test.sh        # Camera test
```

## GPU Cluster Specs

- **GPUs**: 8x AMD MI300X (1.5TB total VRAM)
- **RAM**: 1920GB, **CPU**: 160 vCPU
- **Storage**: 2TB boot + 40TB scratch NVMe
- **Framework**: PyTorch 2.6.0 + ROCm 7.0.0
- **Budget**: ~$200 total (~13 hours at $15.92/hr)

## Training Pipeline

| Stage | Model | Duration | GPUs |
|-------|-------|----------|------|
| Data download + preprocess | All | ~1hr | CPU |
| Synthetic data gen | DeepSeek-R1-70B teacher | ~1hr | 4 |
| LLM Stage 1: Conversational SFT | Qwen3-1.7B | ~2hr | 8 |
| LLM Stage 2: Tool calling | Qwen3-1.7B | ~1hr | 8 |
| LLM Stage 3: Reasoning | Qwen3-1.7B | ~1hr | 8 |
| LLM Stage 4: GRPO | Qwen3-1.7B | ~1hr | 8 |
| LLM quantize | llama.cpp | ~30min | CPU |
| STT training | Whisper Small | ~2hr | 4 |
| TTS training | Piper VITS | ~4hr | 2 |
| VLM training | SmolVLM-500M | ~2hr | 8 |

## Synthetic Data

Generated on the GPU droplet using **DeepSeek-R1-Distill-Llama-70B** (Q4_K_M GGUF):

- **3,000** Hinglish conversation pairs
- **1,000** tool call examples
- **2,000** think/no-think reasoning pairs
- **2,000** home scene VQA pairs

## HuggingFace

- **Repo**: [moudook/VASU_Versatile_AI_System_for_Home_Understanding](https://huggingface.co/moudook/VASU_Versatile_AI_System_for_Home_Understanding)
- Checkpoints pushed every 5 hours via cron
- Final models tagged: `vasu-llm-final-q4`, `vasu-stt-final-onnx`, `vasu-tts-final-onnx`, `vasu-vlm-final`

## Device Specs (Redmi 7A)

- **SoC**: Snapdragon 439, 8x Cortex-A53 @ 1.95GHz
- **RAM**: 2GB LPDDR4X (hard constraint)
- **Storage**: 32GB eMMC
- **OS**: postmarketOS (Alpine Linux)
- **Interface**: Voice only, headless

### RAM Allocation

```
OS + kernel:     ~300MB
STT (Whisper):   ~150MB (always on)
TTS (Piper):     ~80MB  (always on)
Wake word:       ~20MB  (always on)
LLM/VLM slot:    ~1350MB (hot-swap)
```

## Vasu Personality

- Name: Vasu (वासु)
- Language: Hinglish by default, adapts to user
- Responses: Short, direct, no filler
- Adapts: Tech-savvy users get technical answers, family members get simpler explanations
- No corporate chatbot behavior. Answers like a knowledgeable friend.

## License

Personal use only. Not for commercial distribution.
