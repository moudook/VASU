#!/usr/bin/env python3
"""
VASU Project — Autonomous Deployment Script
============================================
Usage:
    export HF_TOKEN="hf_your_token_here"
    python deploy.py --ip <GPU_DROPLET_IP> [--key <ssh_key_path>] [--user <username>]

This script:
1. Uploads the entire remote/ directory to the GPU droplet
2. Runs setup.sh (non-interactive, all deps installed)
3. Sets up the 5-hour HF push cron job
4. Launches master_run.sh inside a tmux session
5. Tails the log so you can watch the first few minutes
6. Detaches — training continues autonomously
"""

import argparse
import os
import subprocess
import sys
import time
import shutil


REMOTE_BASE = "/home/vasu"
REMOTE_DIR = f"{REMOTE_BASE}/remote"
LOG_DIR = f"{REMOTE_BASE}/logs"
SCRATCH_DIR = "/scratch/vasu"


def log(msg: str):
    print(f"[VASU-DEPLOY] {time.strftime('%H:%M:%S')} — {msg}")


def check_prerequisites():
    """Verify local tools exist."""
    for tool in ["ssh", "rsync"]:
        if not shutil.which(tool):
            log(f"ERROR: '{tool}' not found in PATH. Install it first.")
            sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("ERROR: HF_TOKEN environment variable not set.")
        log("       export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)
    log("Prerequisites OK.")
    return hf_token


def build_ssh_cmd(user: str, ip: str, key: str = None):
    """Build base SSH command list."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30"]
    if key:
        cmd += ["-i", key]
    cmd.append(f"{user}@{ip}")
    return cmd


def build_rsync_cmd(user: str, ip: str, key: str = None):
    """Build base rsync command list."""
    cmd = ["rsync", "-avz", "--progress", "--exclude", "__pycache__", "--exclude", "*.pyc"]
    if key:
        cmd += ["-e", f"ssh -i {key} -o StrictHostKeyChecking=no"]
    else:
        cmd += ["-e", "ssh -o StrictHostKeyChecking=no"]
    return cmd


def run_ssh(user: str, ip: str, key: str, command: str, check: bool = True, stream: bool = False):
    """Run a command on the remote machine via SSH."""
    ssh_cmd = build_ssh_cmd(user, ip, key)
    ssh_cmd.append(command)
    log(f"SSH> {command[:120]}{'...' if len(command) > 120 else ''}")

    if stream:
        proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                print(f"  | {line}", end="")
        except KeyboardInterrupt:
            log("Interrupted by user — training continues on droplet.")
            return
        proc.wait()
        if check and proc.returncode != 0:
            log(f"ERROR: Remote command exited with code {proc.returncode}")
            sys.exit(1)
    else:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            log(f"ERROR: {result.stderr.strip()}")
            sys.exit(1)
        return result.stdout.strip()


def upload_files(user: str, ip: str, key: str):
    """Upload the remote/ directory to the droplet."""
    local_remote_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote")
    if not os.path.isdir(local_remote_dir):
        log(f"ERROR: '{local_remote_dir}' directory not found.")
        sys.exit(1)

    # Create remote directories
    run_ssh(user, ip, key, f"mkdir -p {REMOTE_DIR} {LOG_DIR} {SCRATCH_DIR}")

    # rsync upload
    rsync_cmd = build_rsync_cmd(user, ip, key)
    rsync_cmd += [f"{local_remote_dir}/", f"{user}@{ip}:{REMOTE_DIR}/"]

    log("Uploading project files...")
    result = subprocess.run(rsync_cmd, capture_output=False)
    if result.returncode != 0:
        log("ERROR: rsync failed.")
        sys.exit(1)
    log("Upload complete.")


def setup_environment(user: str, ip: str, key: str, hf_token: str):
    """Run setup.sh on the droplet."""
    log("Running setup.sh (this may take 15-30 minutes)...")
    commands = [
        f"chmod +x {REMOTE_DIR}/setup.sh {REMOTE_DIR}/master_run.sh {REMOTE_DIR}/cron_setup.sh",
        f"chmod +x {REMOTE_DIR}/quantize/*.sh",
        f"export HF_TOKEN='{hf_token}' && bash {REMOTE_DIR}/setup.sh 2>&1 | tee {LOG_DIR}/setup.log",
    ]
    for cmd in commands:
        run_ssh(user, ip, key, cmd, stream=True)
    log("Setup complete.")


def setup_cron(user: str, ip: str, key: str, hf_token: str):
    """Set up the 5-hour HF push cron job."""
    log("Setting up HF push cron job...")
    run_ssh(user, ip, key,
            f"export HF_TOKEN='{hf_token}' && bash {REMOTE_DIR}/cron_setup.sh 2>&1 | tee {LOG_DIR}/cron_setup.log",
            stream=True)
    log("Cron job configured.")


def launch_training(user: str, ip: str, key: str, hf_token: str):
    """Launch master_run.sh inside a tmux session."""
    log("Launching training in tmux session 'vasu_training'...")

    # Kill existing session if any
    run_ssh(user, ip, key, "tmux kill-session -t vasu_training 2>/dev/null || true", check=False)

    # Launch new tmux session with HF_TOKEN in environment
    tmux_cmd = (
        f"tmux new-session -d -s vasu_training "
        f"\"export HF_TOKEN='{hf_token}' && "
        f"bash {REMOTE_DIR}/master_run.sh 2>&1 | tee {LOG_DIR}/master_run.log\""
    )
    run_ssh(user, ip, key, tmux_cmd)
    log("Training session launched.")


def tail_logs(user: str, ip: str, key: str, seconds: int = 120):
    """Tail training logs for a few minutes so user can verify start."""
    log(f"Tailing logs for {seconds} seconds (Ctrl+C to detach, training continues)...")
    log("=" * 60)
    try:
        run_ssh(user, ip, key,
                f"timeout {seconds} tail -f {LOG_DIR}/master_run.log 2>/dev/null || "
                f"sleep 5 && timeout {seconds} tail -f {LOG_DIR}/master_run.log",
                stream=True, check=False)
    except KeyboardInterrupt:
        pass
    log("=" * 60)
    log("Detached from logs. Training continues autonomously on the droplet.")
    log(f"Re-attach anytime: ssh {user}@{ip} -t 'tmux attach -t vasu_training'")
    log(f"Check logs: ssh {user}@{ip} 'tail -100 {LOG_DIR}/master_run.log'")


def main():
    parser = argparse.ArgumentParser(description="VASU Project — Autonomous GPU Deployment")
    parser.add_argument("--ip", required=True, help="IP address of the GPU droplet")
    parser.add_argument("--key", default=None, help="Path to SSH private key (optional)")
    parser.add_argument("--user", default="root", help="SSH username (default: root)")
    parser.add_argument("--tail-seconds", type=int, default=120,
                        help="Seconds to tail logs before detaching (default: 120)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip setup.sh (if already installed)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip file upload (if already uploaded)")
    args = parser.parse_args()

    log("=" * 60)
    log("VASU PROJECT — AUTONOMOUS DEPLOYMENT")
    log("=" * 60)

    hf_token = check_prerequisites()

    if not args.skip_upload:
        upload_files(args.user, args.ip, args.key)

    if not args.skip_setup:
        setup_environment(args.user, args.ip, args.key, hf_token)

    setup_cron(args.user, args.ip, args.key, hf_token)
    launch_training(args.user, args.ip, args.key, hf_token)
    tail_logs(args.user, args.ip, args.key, args.tail_seconds)

    log("DEPLOYMENT COMPLETE. All systems autonomous.")


if __name__ == "__main__":
    main()
