#!/bin/bash
###############################################################################
# VASU — GPU Droplet Environment Setup
# Runs non-interactively. Installs all dependencies for training 4 models.
# Target: 1x AMD MI300X, ROCm 7.0.0, PyTorch 2.6.0
# Optimized for 192GB VRAM, 20 vCPU, 240GB RAM
###############################################################################

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LOG_DIR="/home/vasu/logs"
SCRATCH="/scratch/vasu"
mkdir -p "$LOG_DIR" "$SCRATCH"/{datasets,checkpoints,models}

log() {
    echo "[SETUP $(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/setup.log"
}

log "========== VASU SETUP START =========="

# ─────────────────────────────────────────────
# 1. System packages
# ─────────────────────────────────────────────
log "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip python3-venv python3-dev \
    git git-lfs wget curl tmux htop nvtop \
    ffmpeg espeak-ng sox libsox-dev libsndfile1-dev \
    build-essential cmake ninja-build \
    libbz2-dev libffi-dev libssl-dev zlib1g-dev \
    jq bc cron unzip pigz \
    2>&1 | tail -5

git lfs install --skip-smudge 2>/dev/null || true

# ─────────────────────────────────────────────
# 2. Python environment
# ─────────────────────────────────────────────
log "Setting up Python environment..."
python3 -m venv /home/vasu/venv --system-site-packages 2>/dev/null || true
source /home/vasu/venv/bin/activate 2>/dev/null || true

# Ensure pip is up-to-date
pip install --upgrade pip setuptools wheel 2>&1 | tail -3

# ─────────────────────────────────────────────
# 3. PyTorch ROCm (if not pre-installed)
# ─────────────────────────────────────────────
log "Checking PyTorch ROCm..."
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null && {
    log "PyTorch ROCm already installed and GPUs detected."
} || {
    log "Installing PyTorch ROCm 7.1 for MI300X..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm7.1 \
        2>&1 | tail -5
}

# Print GPU info
python3 -c "
import torch
import os
# Enable MI300X optimizations
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'garbage_collection_threshold:0.8,max_split_size_mb:512'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.4.2'

print(f'PyTorch {torch.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
" 2>&1 | tee -a "$LOG_DIR/setup.log"

# ─────────────────────────────────────────────
# 4. ML Libraries
# ─────────────────────────────────────────────
log "Installing ML libraries..."

pip install --no-cache-dir \
    transformers>=4.46.0 \
    datasets>=3.0.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    bitsandbytes \
    sentencepiece \
    protobuf \
    tokenizers \
    safetensors \
    huggingface_hub \
    wandb \
    tqdm \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    librosa \
    soundfile \
    audioread \
    jiwer \
    evaluate \
    tensorboard \
    optimum \
    onnx \
    onnxruntime \
    2>&1 | tail -10

# ─────────────────────────────────────────────
# 5. Unsloth (fast LoRA training)
# ─────────────────────────────────────────────
log "Installing Unsloth..."
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -5 || {
    log "WARNING: Unsloth install failed. Will fall back to standard PEFT training."
}

# ─────────────────────────────────────────────
# 6. llama.cpp (ROCm build for quantization)
# ─────────────────────────────────────────────
log "Building llama.cpp with ROCm support..."
LLAMA_DIR="$SCRATCH/llama.cpp"
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    cd "$SCRATCH"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true
    cd "$LLAMA_DIR"
    mkdir -p build && cd build
    cmake .. \
        -DGGML_HIP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="gfx942" \
        2>&1 | tail -5
    cmake --build . --config Release -j $(nproc) 2>&1 | tail -5
    log "llama.cpp built successfully."
else
    log "llama.cpp already built."
fi

# ─────────────────────────────────────────────
# 7. Piper TTS training dependencies
# ─────────────────────────────────────────────
log "Installing Piper TTS training deps..."
PIPER_DIR="$SCRATCH/piper"
if [ ! -d "$PIPER_DIR" ]; then
    cd "$SCRATCH"
    git clone --depth 1 https://github.com/rhasspy/piper.git 2>/dev/null || true
    cd "$PIPER_DIR"
    pip install --no-cache-dir -e ".[train]" 2>&1 | tail -5 || {
        log "WARNING: Piper install from source failed. Installing deps manually..."
        pip install --no-cache-dir \
            pytorch-lightning \
            piper-phonemize \
            2>&1 | tail -5
    }
else
    log "Piper already cloned."
fi

# Montreal Forced Aligner for TTS
log "Installing Montreal Forced Aligner..."
pip install --no-cache-dir montreal-forced-aligner 2>&1 | tail -3 || {
    log "WARNING: MFA install failed. TTS alignment may need manual setup."
}

# ─────────────────────────────────────────────
# 8. DeepSpeed (optional, for distributed training)
# ─────────────────────────────────────────────
log "Installing DeepSpeed..."
pip install --no-cache-dir deepspeed 2>&1 | tail -3 || {
    log "WARNING: DeepSpeed install failed. Using accelerate distributed instead."
}

# ─────────────────────────────────────────────
# 9. Mount scratch disk if not mounted
# ─────────────────────────────────────────────
log "Checking scratch disk..."
if mountpoint -q /scratch; then
    log "/scratch is mounted. $(df -h /scratch | tail -1)"
else
    log "WARNING: /scratch is not a separate mount. Using it as a directory."
    mkdir -p /scratch/vasu
fi

# ─────────────────────────────────────────────
# 10. Create directory structure
# ─────────────────────────────────────────────
log "Creating directory structure..."
mkdir -p "$SCRATCH"/{datasets,checkpoints,models}/{llm,stt,tts,vlm}
mkdir -p "$SCRATCH"/datasets/llm/{stage1,stage2,stage3,stage4,grpo}
mkdir -p "$SCRATCH"/datasets/synthetic
mkdir -p "$SCRATCH"/models/final

# ─────────────────────────────────────────────
# 11. Verify installation
# ─────────────────────────────────────────────
log "Verifying installation..."
python3 << 'VERIFY'
import sys
errors = []

def check(name):
    try:
        __import__(name)
    except ImportError:
        errors.append(name)

check("torch")
check("transformers")
check("datasets")
check("accelerate")
check("peft")
check("trl")
check("huggingface_hub")
check("librosa")
check("soundfile")
check("evaluate")
check("onnx")

if errors:
    print(f"WARNING: Missing packages: {errors}")
    sys.exit(1)
else:
    print("All core packages verified OK.")

import torch
assert torch.cuda.is_available(), "No GPU detected!"
print(f"GPUs available: {torch.cuda.device_count()}")
VERIFY

log "========== VASU SETUP COMPLETE =========="
