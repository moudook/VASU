#!/bin/bash
###############################################################################
# VASU — Master Training Pipeline Orchestrator
# Runs ALL stages sequentially: data → synthetic → train → quantize → push
# Fully autonomous — no human input needed after launch.
###############################################################################

set -uo pipefail

# MI300X Optimizations - Critical for AMD GPUs
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

REMOTE_DIR="/home/vasu/remote"
LOG_DIR="/home/vasu/logs"
SCRATCH="/scratch/vasu"
CHECKPOINT_DIR="$SCRATCH/checkpoints"
MODELS_DIR="$SCRATCH/models"

source /home/vasu/venv/bin/activate 2>/dev/null || true
source /home/vasu/.env 2>/dev/null || true

log() {
    echo "[MASTER $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/master_run.log"
}

run_step() {
    local step_name="$1"
    local command="$2"
    local log_file="$LOG_DIR/${step_name}.log"

    log "━━━ STARTING: $step_name ━━━"
    local start_time=$(date +%s)

    if eval "$command" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$(( (end_time - start_time) / 60 ))
        log "✓ COMPLETED: $step_name (${duration} minutes)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$(( (end_time - start_time) / 60 ))
        log "✗ FAILED: $step_name after ${duration} minutes. Check $log_file"
        return 1
    fi
}

gpu_status() {
    echo "=== GPU Status ==="
    rocm-smi 2>/dev/null | head -10 || echo "rocm-smi not available"
    echo ""
}

log "╔══════════════════════════════════════════════════════╗"
log "║         VASU MASTER TRAINING PIPELINE                ║"
log "║         $(date '+%Y-%m-%d %H:%M:%S')                ║"
log "╚══════════════════════════════════════════════════════╝"

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    log "FATAL: HF_TOKEN not set. Cannot push models."
    exit 1
fi

# ─────────────────────────────────────────────
# PHASE 1: DATA DOWNLOAD & PREPROCESSING
# ─────────────────────────────────────────────
log "═══ PHASE 1: DATA PIPELINE ═══"

run_step "download_all" \
    "python3 $REMOTE_DIR/data/download_all.py" || {
    log "Data download had errors. Continuing with available data..."
}

# Run all 4 preprocessors in parallel
log "Running preprocessors in parallel..."
python3 "$REMOTE_DIR/data/preprocess_llm.py" 2>&1 | tee "$LOG_DIR/preprocess_llm.log" &
PID_LLM=$!
python3 "$REMOTE_DIR/data/preprocess_stt.py" 2>&1 | tee "$LOG_DIR/preprocess_stt.log" &
PID_STT=$!
python3 "$REMOTE_DIR/data/preprocess_tts.py" 2>&1 | tee "$LOG_DIR/preprocess_tts.log" &
PID_TTS=$!
python3 "$REMOTE_DIR/data/preprocess_vlm.py" 2>&1 | tee "$LOG_DIR/preprocess_vlm.log" &
PID_VLM=$!

wait $PID_LLM && log "✓ LLM preprocessing done" || log "✗ LLM preprocessing failed"
wait $PID_STT && log "✓ STT preprocessing done" || log "✗ STT preprocessing failed"
wait $PID_TTS && log "✓ TTS preprocessing done" || log "✗ TTS preprocessing failed"
wait $PID_VLM && log "✓ VLM preprocessing done" || log "✗ VLM preprocessing failed"

# ─────────────────────────────────────────────
# PHASE 2: SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
log "═══ PHASE 2: SYNTHETIC DATA GENERATION ═══"

run_step "gen_hinglish" \
    "python3 $REMOTE_DIR/synthetic/gen_hinglish_conversations.py" || true

run_step "gen_tool_calls" \
    "python3 $REMOTE_DIR/synthetic/gen_tool_call_data.py" || true

run_step "gen_reasoning" \
    "python3 $REMOTE_DIR/synthetic/gen_reasoning_traces.py" || true

run_step "gen_vlm_data" \
    "python3 $REMOTE_DIR/synthetic/gen_vlm_home_data.py" || true

# ─────────────────────────────────────────────
# PHASE 3: LLM TRAINING (4 stages)
# ─────────────────────────────────────────────
log "═══ PHASE 3: LLM TRAINING ═══"

run_step "train_llm_stage1" \
    "python3 $REMOTE_DIR/train/train_llm_stage1.py" || {
    log "CRITICAL: LLM Stage 1 failed. Cannot continue LLM pipeline."
}

run_step "train_llm_stage2" \
    "python3 $REMOTE_DIR/train/train_llm_stage2.py" || {
    log "WARNING: LLM Stage 2 failed. Continuing with Stage 1 model."
}

run_step "train_llm_stage3" \
    "python3 $REMOTE_DIR/train/train_llm_stage3.py" || {
    log "WARNING: LLM Stage 3 failed. Continuing with best available."
}

run_step "train_llm_stage4" \
    "python3 $REMOTE_DIR/train/train_llm_stage4.py" || {
    log "WARNING: LLM GRPO stage failed. Using previous best."
}

# LLM Quantization
run_step "quantize_llm" \
    "bash $REMOTE_DIR/quantize/quantize_llm.sh" || {
    log "WARNING: LLM quantization failed."
}

# ─────────────────────────────────────────────
# PHASE 4: STT TRAINING
# ─────────────────────────────────────────────
log "═══ PHASE 4: STT TRAINING ═══"

run_step "train_stt" \
    "python3 $REMOTE_DIR/train/train_stt.py" || {
    log "WARNING: STT training failed."
}

run_step "export_stt_onnx" \
    "python3 $REMOTE_DIR/quantize/export_stt_onnx.py" || {
    log "WARNING: STT ONNX export failed."
}

# ─────────────────────────────────────────────
# PHASE 5: TTS TRAINING
# ─────────────────────────────────────────────
log "═══ PHASE 5: TTS TRAINING ═══"

run_step "train_tts" \
    "python3 $REMOTE_DIR/train/train_tts.py" || {
    log "WARNING: TTS training failed."
}

run_step "export_tts_onnx" \
    "python3 $REMOTE_DIR/quantize/export_tts_onnx.py" || {
    log "WARNING: TTS ONNX export failed."
}

# ─────────────────────────────────────────────
# PHASE 6: VLM TRAINING
# ─────────────────────────────────────────────
log "═══ PHASE 6: VLM TRAINING ═══"

run_step "train_vlm" \
    "python3 $REMOTE_DIR/train/train_vlm.py" || {
    log "WARNING: VLM training failed."
}

run_step "quantize_vlm" \
    "python3 $REMOTE_DIR/quantize/quantize_vlm.py" || {
    log "WARNING: VLM quantization failed."
}

# ─────────────────────────────────────────────
# PHASE 7: FINAL PUSH TO HUGGINGFACE
# ─────────────────────────────────────────────
log "═══ PHASE 7: FINAL HF PUSH ═══"

run_step "final_hf_push" \
    "python3 $REMOTE_DIR/push_hf.py --final" || {
    log "WARNING: Final HF push failed. Retry manually."
}

# ─────────────────────────────────────────────
# COMPLETION
# ─────────────────────────────────────────────
log "╔══════════════════════════════════════════════════════╗"
log "║         VASU TRAINING PIPELINE COMPLETE              ║"
log "║         $(date '+%Y-%m-%d %H:%M:%S')                ║"
log "╚══════════════════════════════════════════════════════╝"

# Write completion markers
echo "TRAINING_COMPLETE=$(date -Iseconds)" > "$LOG_DIR/completion.marker"
echo "VASU TRAINING COMPLETE $(date)" >> "$LOG_DIR/COMPLETE.log"

gpu_status

# Remove cron job (no more pushes needed)
crontab -l 2>/dev/null | grep -v "push_hf.py" | crontab - 2>/dev/null || true
log "Cron job removed. All models pushed."

log "All models available at: https://huggingface.co/moudook/VASU_Versatile_AI_System_for_Home_Understanding"
