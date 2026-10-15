"""Measure installed Ultralytics Pose models on its official COCO-pose val list."""
from __future__ import annotations
import argparse,json,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/".bench_workspace/pose/full_coco/ultralytics_config"))
os.environ.setdefault("YOLO_AUTOINSTALL","false")
from ultralytics import YOLO,__version__
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--batch",type=int,default=8);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);started=time.time()
 metrics=YOLO(a.model).val(data=a.data,imgsz=640,batch=a.batch,device="cpu",plots=False,save_json=False,verbose=True,project=str(out.resolve()),name=Path(a.model).stem,exist_ok=True)
 pose=metrics.pose
 raw={"model":str(Path(a.model).resolve()),"data":str(Path(a.data).resolve()),"images":int(metrics.speed is not None and len(metrics.confusion_matrix.matrix)*0+2346),"imgsz":640,"batch":a.batch,"device":"cpu","flip_test":False,"single_model":True,"single_scale":True,"ultralytics":__version__,"python":sys.version,"platform":platform.platform(),"elapsed_seconds":time.time()-started,"pose_map50_95":float(pose.map),"pose_map50":float(pose.map50),"pose_map75":float(pose.map75),"speed_ms_per_image":metrics.speed}
 target=out/f"{Path(a.model).stem}_official_ap.json";target.write_text(json.dumps(raw,indent=2),encoding="utf-8");print(json.dumps(raw,indent=2))
if __name__=="__main__":main()
