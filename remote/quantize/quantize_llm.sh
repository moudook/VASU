#!/bin/bash
###############################################################################
# VASU — LLM Quantization: Convert to GGUF Q4_K_M + Q5_K_M
# Uses llama.cpp built with ROCm support
###############################################################################

set -euo pipefail

LOG="/home/vasu/logs/quantize_llm.log"
SCRATCH="/scratch/vasu"
MODEL_DIR="$SCRATCH/models/llm/stage4"
LLAMA_CPP="$SCRATCH/llama.cpp"
OUTPUT_DIR="$SCRATCH/models/final"

log() {
    echo "[QUANTIZE-LLM $(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log "═══ VASU LLM QUANTIZATION ═══"

# Find the best available model (stage4 > stage3 > stage2 > stage1)
for stage in stage4 stage3 stage2 stage1; do
    candidate="$SCRATCH/models/llm/$stage"
    if [ -d "$candidate" ] && [ "$(ls -A $candidate 2>/dev/null)" ]; then
        MODEL_DIR="$candidate"
        log "Using model: $MODEL_DIR"
        break
    fi
done

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    log "ERROR: No trained LLM model found!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Convert to FP16 GGUF
log "Converting to FP16 GGUF..."
CONVERT_SCRIPT="$LLAMA_CPP/convert_hf_to_gguf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    CONVERT_SCRIPT="$LLAMA_CPP/convert-hf-to-gguf.py"
fi

if [ ! -f "$CONVERT_SCRIPT" ]; then
    log "ERROR: convert script not found in $LLAMA_CPP"
    exit 1
fi

FP16_GGUF="$OUTPUT_DIR/vasu_llm_fp16.gguf"
if [ ! -f "$FP16_GGUF" ]; then
    python3 "$CONVERT_SCRIPT" "$MODEL_DIR" \
        --outfile "$FP16_GGUF" \
        --outtype f16 \
        2>&1 | tee -a "$LOG"
    log "✓ FP16 GGUF created: $FP16_GGUF"
else
    log "FP16 GGUF already exists."
fi

# Step 2: Quantize to Q4_K_M
log "Quantizing to Q4_K_M..."
QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="$LLAMA_CPP/build/bin/quantize"
fi

Q4_GGUF="$OUTPUT_DIR/vasu_llm_q4_k_m.gguf"
if [ ! -f "$Q4_GGUF" ]; then
    "$QUANTIZE_BIN" "$FP16_GGUF" "$Q4_GGUF" Q4_K_M \
        2>&1 | tee -a "$LOG"
    log "✓ Q4_K_M: $Q4_GGUF ($(du -h "$Q4_GGUF" | cut -f1))"
else
    log "Q4_K_M already exists."
fi

# Step 3: Quantize to Q5_K_M (for quality comparison)
log "Quantizing to Q5_K_M..."
Q5_GGUF="$OUTPUT_DIR/vasu_llm_q5_k_m.gguf"
if [ ! -f "$Q5_GGUF" ]; then
    "$QUANTIZE_BIN" "$FP16_GGUF" "$Q5_GGUF" Q5_K_M \
        2>&1 | tee -a "$LOG"
    log "✓ Q5_K_M: $Q5_GGUF ($(du -h "$Q5_GGUF" | cut -f1))"
else
    log "Q5_K_M already exists."
fi

# Step 4: Quick benchmark on CPU
log "Running CPU inference benchmark..."
MAIN_BIN="$LLAMA_CPP/build/bin/llama-cli"
if [ ! -f "$MAIN_BIN" ]; then
    MAIN_BIN="$LLAMA_CPP/build/bin/main"
fi

if [ -f "$MAIN_BIN" ]; then
    "$MAIN_BIN" -m "$Q4_GGUF" \
        -p "Hello, I am Vasu. " \
        -n 50 \
        --threads 4 \
        --no-display-prompt \
        2>&1 | grep -E "eval|token" | tee -a "$LOG" || true
    log "Benchmark complete."
else
    log "WARNING: llama-cli not found. Skipping benchmark."
fi

# Cleanup FP16 to save space
log "Cleaning up FP16 intermediate..."
rm -f "$FP16_GGUF"

log "═══ LLM QUANTIZATION COMPLETE ═══"
log "Q4_K_M: $Q4_GGUF"
log "Q5_K_M: $Q5_GGUF"
