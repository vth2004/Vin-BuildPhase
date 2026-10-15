"""Measure cold-process RSS and sequential end-to-end latency for one mode."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/"experiments/model_check";sys.path.insert(0,str(OLD));WORK=ROOT/".bench_workspace/pose";os.environ["XDG_CACHE_HOME"]=str(WORK/"models/xdg_cache")
from models_manager import YoloPoseModel,RTMPoseModel
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["K1","K2","K3"]);p.add_argument("--output",required=True);a=p.parse_args();models=[YoloPoseModel("yolo26s-pose",ROOT/"backend/yolo26s-pose.pt")]
 if a.mode in ("K2","K3"):models.append(RTMPoseModel("balanced"))
 if a.mode=="K3":models.append(YoloPoseModel("yolov8m-pose",WORK/"models/yolov8m-pose.pt"))
 man=json.loads((WORK/"data/split_manifest.json").read_text());imgs=[WORK/"data/images"/x["file_name"] for x in man["test"][:35]];proc=psutil.Process();peak=proc.memory_info().rss;lat=[]
 for i,img in enumerate(imgs):
  t=time.perf_counter()
  for m in models:m.predict(img);peak=max(peak,proc.memory_info().rss)
  if i>=5:lat.append((time.perf_counter()-t)*1000)
 r={"mode":a.mode,"images_measured":30,"latency_median_ms":float(np.median(lat)),"latency_p95_ms":float(np.percentile(lat,95)),"peak_rss_mb":peak/1048576}
 Path(a.output).write_text(json.dumps(r,indent=2),encoding="utf-8");print(json.dumps(r))
if __name__=="__main__":main()
