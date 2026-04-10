#!/bin/bash
###############################################################################
# VASU - Redmi 7A (mi439) Device Setup
# Full setup from fresh postmarketOS / Alpine Linux install.
# Installs all dependencies, copies models, enables services.
###############################################################################

set -euo pipefail

LOG="/var/vasu/logs/setup_device.log"
mkdir -p /var/vasu/{logs,notes} /etc/vasu /opt/vasu/{models,orchestrator,bin,sounds}

log() {
    echo "[DEVICE-SETUP $(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log "========== VASU DEVICE SETUP START =========="

# ---- 1. System Packages ----
log "Installing system packages..."
apk update
apk add \
    python3 py3-pip py3-numpy py3-pillow \
    alsa-utils alsa-lib \
    libcamera libcamera-tools \
    ffmpeg espeak-ng \
    git wget curl \
    build-base cmake \
    linux-firmware \
    sox \
    2>&1 | tail -5

# ---- 2. Python Dependencies ----
log "Installing Python packages..."
pip3 install --no-cache-dir \
    llama-cpp-python \
    onnxruntime \
    librosa \
    soundfile \
    pyaudio \
    openwakeword \
    transformers \
    Pillow \
    requests \
    2>&1 | tail -5

# ---- 3. Piper TTS Binary ----
log "Installing Piper TTS..."
PIPER_VERSION="2023.11.14-2"
PIPER_ARCH="armv7"  # Redmi 7A is 32-bit ARM
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"

if [ ! -f /opt/vasu/bin/piper ]; then
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz 2>/dev/null || {
        # Try aarch64 if armv7 not available
        PIPER_ARCH="aarch64"
        PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"
        wget -q "$PIPER_URL" -O /tmp/piper.tar.gz
    }
    tar -xzf /tmp/piper.tar.gz -C /opt/vasu/bin/
    rm /tmp/piper.tar.gz
    log "Piper installed"
else
    log "Piper already installed"
fi

# ---- 4. Whisper.cpp (ARM NEON optimized) ----
log "Building whisper.cpp..."
if [ ! -f /opt/vasu/bin/whisper-cpp ]; then
    cd /tmp
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git 2>/dev/null || true
    cd whisper.cpp
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build . --config Release -j $(nproc) 2>&1 | tail -3
    cp bin/whisper-cli /opt/vasu/bin/whisper-cpp 2>/dev/null || \
        cp bin/main /opt/vasu/bin/whisper-cpp 2>/dev/null || true
    cd / && rm -rf /tmp/whisper.cpp
    log "whisper.cpp built"
else
    log "whisper.cpp already present"
fi

# ---- 5. llama.cpp (ARM NEON) ----
log "Building llama.cpp..."
if [ ! -f /opt/vasu/bin/llama-server ]; then
    cd /tmp
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true
    cd llama.cpp
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build . --config Release -j $(nproc) 2>&1 | tail -3
    cp bin/llama-cli /opt/vasu/bin/ 2>/dev/null || true
    cp bin/llama-server /opt/vasu/bin/ 2>/dev/null || true
    cd / && rm -rf /tmp/llama.cpp
    log "llama.cpp built"
else
    log "llama.cpp already present"
fi

# ---- 6. Copy Orchestrator Code ----
log "Copying orchestrator files..."
# Assumes orchestrator files are on a USB drive or downloaded
if [ -d /media/usb/orchestrator ]; then
    cp -r /media/usb/orchestrator/* /opt/vasu/orchestrator/
elif [ -d /tmp/vasu_deploy/orchestrator ]; then
    cp -r /tmp/vasu_deploy/orchestrator/* /opt/vasu/orchestrator/
fi

# ---- 7. Copy Models ----
log "Checking for models..."
for model in vasu_llm.gguf vasu_stt.onnx vasu_tts.onnx; do
    if [ -f /opt/vasu/models/$model ]; then
        log "  Found: $model"
    else
        log "  MISSING: $model — download from HuggingFace"
    fi
done

if [ -d /opt/vasu/models/vasu_vlm ]; then
    log "  Found: vasu_vlm/"
else
    log "  MISSING: vasu_vlm/ — download from HuggingFace"
fi

# ---- 8. Create Default Config ----
log "Creating default config..."
cat > /etc/vasu/config.json << 'CONFIG'
{
    "wake_word": "hey_vasu",
    "language": "hi",
    "max_response_tokens": 256,
    "llm_model": "vasu_llm.gguf",
    "stt_model": "vasu_stt.onnx",
    "tts_model": "vasu_tts.onnx",
    "vlm_model_dir": "vasu_vlm",
    "silence_timeout_sec": 2.0,
    "thermal_throttle_temp": 70,
    "thermal_pause_temp": 80
}
CONFIG

# ---- 9. Install Systemd Services ----
log "Installing systemd services..."
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [ -d "$SCRIPT_DIR/systemd" ]; then
    cp "$SCRIPT_DIR/systemd/"*.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable vasu-wake.service
    systemctl enable vasu-stt.service
    systemctl enable vasu-tts.service
    systemctl enable vasu-orchestrator.service
    log "Services installed and enabled"
else
    log "WARNING: systemd service files not found"
fi

# ---- 10. CPU Governor Defaults ----
log "Setting CPU governor defaults..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo powersave > "$cpu" 2>/dev/null || true
done

# ---- 11. Cgroups Setup ----
log "Setting up cgroups..."
if [ -d /sys/fs/cgroup/memory ]; then
    for group in vasu_wake vasu_stt vasu_tts vasu_llm; do
        mkdir -p "/sys/fs/cgroup/memory/$group" 2>/dev/null || true
    done
fi

# ---- 12. Create Alarm Sound ----
log "Creating alarm sound..."
# Generate a simple alarm beep using sox
sox -n /opt/vasu/sounds/alarm.wav synth 3 sine 800 sine 1000 \
    remix - fade 0.1 3 0.1 2>/dev/null || {
    # Fallback: generate with python
    python3 -c "
import struct, math, wave
sr = 22050
dur = 3
f = wave.open('/opt/vasu/sounds/alarm.wav', 'wb')
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(sr)
for i in range(sr * dur):
    v = int(32767 * 0.5 * math.sin(2 * math.pi * 800 * i / sr))
    f.writeframes(struct.pack('h', v))
f.close()
"
}

# ---- 13. Test Hardware ----
log "Testing hardware..."

# Camera test
log "  Camera:"
libcamera-hello --list-cameras 2>&1 | tee -a "$LOG" || \
    log "  WARNING: libcamera not working"

# Microphone test
log "  Microphone:"
arecord -l 2>&1 | head -5 | tee -a "$LOG" || \
    log "  WARNING: No recording devices found"

# Speaker test
log "  Speaker:"
aplay -l 2>&1 | head -5 | tee -a "$LOG" || \
    log "  WARNING: No playback devices found"

# ---- 14. Summary ----
log "========== DEVICE SETUP SUMMARY =========="
log "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
log "Kernel: $(uname -r)"
log "RAM: $(free -m | awk '/Mem:/ {print $2}') MB total"
log "Storage: $(df -h / | awk 'NR==2 {print $4}') available"
log "CPU: $(nproc) cores"
log ""
log "Files:"
for f in /opt/vasu/models/*; do
    if [ -e "$f" ]; then
        log "  $(du -h "$f" | cut -f1) $f"
    fi
done
log ""
log "Services:"
systemctl list-unit-files | grep vasu | tee -a "$LOG" || true
log ""
log "========== SETUP COMPLETE =========="
log "Start Vasu: systemctl start vasu-orchestrator"
