"""Evaluate cached real predictions with the existing matching/OKS framework."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/"experiments/model_check";sys.path.insert(0,str(OLD));sys.path.insert(0,str(ROOT))
WORK=ROOT/".bench_workspace/pose"
import evaluate_benchmark as e
e.DATA_DIR=WORK/"data";e.CACHE_DIR=WORK/"cache";e.RESULTS_DIR=WORK/"results";e.RESULTS_DIR.mkdir(parents=True,exist_ok=True)
if __name__=="__main__":e.run_full_benchmark()
