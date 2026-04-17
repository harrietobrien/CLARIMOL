#!/usr/bin/env python3
"""
GPU temperature and utilization monitor.
Logs to file and prints warnings when temperature exceeds threshold.

Usage:
    python scripts/gpu_monitor.py --interval 30 --warn-temp 70 --log-file output/gpu_temps.log
    python scripts/gpu_monitor.py --interval 10 --warn-temp 65  # stdout only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def query_gpu():
    """Query nvidia-smi for temp, utilization, memory, power."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temp_c": int(parts[2]),
                        "util_pct": int(parts[3]),
                        "mem_used_mb": int(parts[4]),
                        "mem_total_mb": int(parts[5]),
                        "power_w": float(parts[6]),
                    }
                )
        return gpus
    except Exception as e:
        print(f"[gpu_monitor] nvidia-smi error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="GPU temperature monitor")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--warn-temp", type=int, default=70, help="Warning threshold (Celsius)")
    parser.add_argument("--crit-temp", type=int, default=85, help="Critical threshold (Celsius)")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    args = parser.parse_args()

    log_fh = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "a")
        log_fh.write(f"# GPU monitor started at {datetime.now().isoformat()}\n")
        log_fh.write("# timestamp,gpu_idx,temp_c,util_pct,mem_used_mb,mem_total_mb,power_w\n")

    print(f"[gpu_monitor] Polling every {args.interval}s "
          f"| warn={args.warn_temp}C | crit={args.crit_temp}C")

    try:
        while True:
            gpus = query_gpu()
            if gpus is None:
                time.sleep(args.interval)
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for gpu in gpus:
                line = (f"{ts},{gpu['index']},{gpu['temp_c']},{gpu['util_pct']},"
                        f"{gpu['mem_used_mb']},{gpu['mem_total_mb']},{gpu['power_w']:.1f}")
                if log_fh:
                    log_fh.write(line + "\n")
                    log_fh.flush()

                if gpu["temp_c"] >= args.crit_temp:
                    msg = (f"[gpu_monitor] CRITICAL: GPU {gpu['index']} at "
                           f"{gpu['temp_c']}C (>{args.crit_temp}C threshold)")
                    print(msg, file=sys.stderr)
                elif gpu["temp_c"] >= args.warn_temp:
                    msg = (f"[gpu_monitor] WARNING: GPU {gpu['index']} at "
                           f"{gpu['temp_c']}C (>{args.warn_temp}C threshold)")
                    print(msg, file=sys.stderr)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[gpu_monitor] Stopped.")
    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
