"""Run real Pose inference on the fixed CrowdPose subset, with isolated cache."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/"experiments/model_check";sys.path[:0]=[str(ROOT/"backend"),str(OLD),str(ROOT)]
BASE=ROOT/".bench_workspace/pose";WORK=BASE/"crowdpose/benchmark";MODELS=BASE/"models"
os.environ["XDG_CACHE_HOME"]=str(MODELS/"xdg_cache");os.chdir(MODELS)
import models_manager as mm
mm.MODELS_DIR=MODELS;mm.LOGS_DIR=WORK/"logs"
import run_inference as ri
ri.DATA_DIR=WORK/"data";ri.IMAGES_DIR=BASE/"crowdpose/images";ri.CACHE_DIR=WORK/"cache";ri.RESULTS_DIR=WORK/"results"
import app.features.human_pose.services.run as prod_run
def main():
 ri.CACHE_DIR.mkdir(parents=True,exist_ok=True);ri.RESULTS_DIR.mkdir(parents=True,exist_ok=True);count={"make_mock_results_calls":0};original=prod_run.make_mock_results
 def counted(*args,**kwargs):count["make_mock_results_calls"]+=1;return original(*args,**kwargs)
 prod_run.make_mock_results=counted
 try:ri.main()
 finally:prod_run.make_mock_results=original
 (ri.RESULTS_DIR/"mock_fallback_audit.json").write_text(json.dumps(count,indent=2),encoding="utf-8")
 if count["make_mock_results_calls"]:raise SystemExit("INVALID benchmark: mock fallback called")
if __name__=="__main__":main()
