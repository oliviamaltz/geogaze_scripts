#!/usr/bin/env python3
"""
run_grid.py — launch 24 training runs for pair groups × mask side.

- Uses your existing training script (edit TRAIN_SCRIPT if needed).
- Forces: --arch resnet50.a1_in1k, --epochs 100
- Keeps all other flags at their defaults.
- Logs each run to logs/<timestamp>_resnet50_mask<side>_<pair>.log
"""

import os
import sys
import subprocess
from datetime import datetime

# --- EDIT THIS IF YOUR FILE IS NAMED DIFFERENTLY OR LIVES ELSEWHERE ---
TRAIN_SCRIPT = "geogaze_training.py"   

PAIR_GROUPS = [
    "bc_bs", "bc_gc", "bc_gs",
    "bs_bc", "bs_gc", "bs_gs",
    "gc_bc", "gc_bs", "gc_gs",
    "gs_bc", "gs_bs", "gs_gc",
]
SIDES = ["L", "R"]

ARCH = "resnet50.a1_in1k"
EPOCHS = "100"
POS_WEIGHT = "100"
THRESHOLD  = "0.3"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_one(pair_mid: str, side: str) -> int:
    """
    Launch a single training run and tee stdout/stderr to a log file.
    Returns the process return code.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{ts}_resnet50_mask{side}_{pair_mid.replace(',', '-').replace('/', '-')}.log"
    log_path = os.path.join(LOG_DIR, log_name)

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--arch", ARCH,
        "--epochs", EPOCHS,
        "--pair_mids", pair_mid,
        "--mask_side", side,
        "--pos_weight", POS_WEIGHT,   
        "--threshold", THRESHOLD,
    ]

    print(f"\n==> Starting: pair_mids={pair_mid} | mask_side={side} | arch={ARCH} | epochs={EPOCHS}")
    print(f"    Logging to: {log_path}")
    print(f"    Cmd: {' '.join(cmd)}")

    with open(log_path, "w", buffering=1) as logf:  # line-buffered
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        ret = proc.wait()

    status = "OK" if ret == 0 else f"FAIL({ret})"
    print(f"<== Finished: pair_mids={pair_mid} | mask_side={side} -> {status}")
    return ret

def main():
    # Sanity checks
    if not os.path.isfile(TRAIN_SCRIPT):
        print(f"ERROR: TRAIN_SCRIPT not found: {TRAIN_SCRIPT}")
        sys.exit(2)

    failures = []
    total = 0

    for pair in PAIR_GROUPS:
        for side in SIDES:
            total += 1
            ret = run_one(pair, side)
            if ret != 0:
                failures.append((pair, side, ret))

    print("\n=== Sweep Summary ===")
    print(f"Total runs: {total}")
    print(f"Successes : {total - len(failures)}")
    print(f"Failures  : {len(failures)}")
    if failures:
        print("Failed combos:")
        for pair, side, code in failures:
            print(f"  - pair_mids={pair}, mask_side={side}, exit_code={code}")
    else:
        print("All runs completed successfully.")

if __name__ == "__main__":
    main()
