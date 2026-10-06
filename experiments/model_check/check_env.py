import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import platform
import psutil
import json
from pathlib import Path

def get_env_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_max_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "python_version": sys.version,
        "executable": sys.executable,
    }
    
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "env_info.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== CAU HINH HE THONG KIEM CHUNG SANDBOX ===\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    
    print(json.dumps(info, indent=2))
    print(f"\n[OK] Log saved to: {log_file}")

if __name__ == "__main__":
    get_env_info()
