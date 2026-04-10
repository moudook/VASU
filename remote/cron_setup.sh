#!/bin/bash
###############################################################################
# VASU — Cron Setup for 5-hour HuggingFace Push
###############################################################################

set -euo pipefail

LOG_DIR="/home/vasu/logs"
REMOTE_DIR="/home/vasu/remote"

log() {
    echo "[CRON-SETUP $(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/cron_setup.log"
}

log "Setting up HuggingFace push cron job..."

# Get current HF_TOKEN
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"

# Create the cron wrapper script
cat > /home/vasu/run_hf_push.sh << CRONSCRIPT
#!/bin/bash
source /home/vasu/venv/bin/activate 2>/dev/null || true
export HF_TOKEN="$HF_TOKEN"
python3 $REMOTE_DIR/push_hf.py --checkpoints-only >> $LOG_DIR/cron_push.log 2>&1
CRONSCRIPT

chmod +x /home/vasu/run_hf_push.sh

# Remove existing vasu cron entries
crontab -l 2>/dev/null | grep -v "run_hf_push" | crontab - 2>/dev/null || true

# Add new cron job: every 5 hours
(crontab -l 2>/dev/null; echo "0 */5 * * * /home/vasu/run_hf_push.sh") | crontab -

# Verify
log "Current crontab:"
crontab -l 2>/dev/null | tee -a "$LOG_DIR/cron_setup.log"

# Start cron service
service cron start 2>/dev/null || systemctl start cron 2>/dev/null || {
    log "WARNING: Could not start cron service. Starting crond..."
    crond 2>/dev/null || true
}

log "Cron job configured: HF push every 5 hours."
