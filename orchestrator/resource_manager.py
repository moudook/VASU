#!/usr/bin/env python3
"""
VASU - Resource Manager
CPU governor control, thermal protection, RAM monitoring, cgroups setup.
Manages system resources on Redmi 7A (Snapdragon 439).
"""

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ResourceManager:
    """System resource management for constrained ARM device."""

    THERMAL_ZONES = "/sys/class/thermal"
    CPU_FREQ_BASE = "/sys/devices/system/cpu"
    MEMINFO = "/proc/meminfo"

    def __init__(self):
        self.num_cpus = os.cpu_count() or 8

    # ---- CPU Governor ----

    def set_power_mode(self, mode):
        """Set CPU governor: 'powersave' or 'performance'."""
        for i in range(self.num_cpus):
            gov_path = os.path.join(
                self.CPU_FREQ_BASE,
                "cpu%d" % i,
                "cpufreq",
                "scaling_governor",
            )
            try:
                Path(gov_path).write_text(mode)
            except (PermissionError, FileNotFoundError):
                pass
        log.debug("CPU governor set to: %s", mode)

    # ---- Thermal Monitoring ----

    def get_temperature(self):
        """Get max CPU temperature in millidegrees."""
        max_temp = 0
        thermal_base = Path(self.THERMAL_ZONES)
        if not thermal_base.exists():
            return 50000  # Default 50C if sysfs not available

        for zone in thermal_base.iterdir():
            temp_file = zone / "temp"
            if temp_file.exists():
                try:
                    temp = int(temp_file.read_text().strip())
                    max_temp = max(max_temp, temp)
                except (ValueError, PermissionError):
                    pass
        return max_temp

    def wait_for_cooldown(self, target_temp=70000, timeout_sec=60):
        """Wait until temperature drops below target."""
        start = time.time()
        while time.time() - start < timeout_sec:
            temp = self.get_temperature()
            if temp < target_temp:
                log.info("Cooled down to %d C", temp // 1000)
                return True
            time.sleep(2)
        log.warning("Cooldown timeout after %d sec", timeout_sec)
        return False

    def should_throttle(self):
        """Check if inference should be throttled due to heat."""
        temp = self.get_temperature()
        if temp > 80000:
            return "pause"  # Too hot, pause inference
        elif temp > 70000:
            return "throttle"  # Hot, add delay between tokens
        return "normal"

    # ---- RAM Monitoring ----

    def get_free_ram_mb(self):
        """Get available RAM in MB from /proc/meminfo."""
        try:
            with open(self.MEMINFO) as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
        return 500  # Default assumption

    def get_used_ram_mb(self):
        """Get used RAM in MB."""
        try:
            with open(self.MEMINFO) as f:
                total = available = 0
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1])
                return (total - available) // 1024
        except Exception:
            return 1500

    def wait_for_ram(self, min_free_mb=800, timeout_sec=5):
        """Wait until enough RAM is available."""
        start = time.time()
        while time.time() - start < timeout_sec:
            free = self.get_free_ram_mb()
            if free >= min_free_mb:
                return True
            log.info("Waiting for RAM: %d MB free, need %d MB", free, min_free_mb)
            import gc
            gc.collect()
            time.sleep(0.5)
        log.warning("RAM wait timeout. Available: %d MB", self.get_free_ram_mb())
        return False

    # ---- Cgroups Setup ----

    def setup_cgroups(self):
        """Set up cgroup memory limits for Vasu processes."""
        cgroup_base = "/sys/fs/cgroup"

        groups = {
            "vasu_wake": {"memory_min": 50 * 1024 * 1024, "oom_score": -1000},
            "vasu_stt": {"memory_min": 200 * 1024 * 1024, "oom_score": -500},
            "vasu_tts": {"memory_min": 100 * 1024 * 1024, "oom_score": -500},
            "vasu_llm": {"memory_min": 0, "oom_score": 0},
        }

        for name, config in groups.items():
            group_path = os.path.join(cgroup_base, "memory", name)
            try:
                os.makedirs(group_path, exist_ok=True)

                # Set memory.min (minimum guaranteed)
                min_path = os.path.join(group_path, "memory.min")
                if os.path.exists(min_path):
                    Path(min_path).write_text(str(config["memory_min"]))

                log.debug("Cgroup configured: %s", name)
            except (PermissionError, FileNotFoundError, OSError) as e:
                log.debug("Cgroup setup skipped for %s: %s", name, e)

    def set_process_priority(self, pid, nice_value):
        """Set process nice value."""
        try:
            os.setpriority(os.PRIO_PROCESS, pid, nice_value)
        except (PermissionError, OSError) as e:
            log.debug("Could not set priority for PID %d: %s", pid, e)

    def set_oom_score(self, pid, score):
        """Set OOM score adjustment for a process."""
        try:
            Path("/proc/%d/oom_score_adj" % pid).write_text(str(score))
        except (PermissionError, FileNotFoundError):
            pass
