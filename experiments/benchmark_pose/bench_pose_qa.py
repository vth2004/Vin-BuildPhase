"""QA-focused K1/K2/K3 benchmark on the fixed 500-image COCO split."""
from __future__ import annotations
import argparse,copy,csv,json,math,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/"experiments/model_check";sys.path.insert(0,str(OLD));sys.path.insert(0,str(ROOT))
import evaluate_benchmark as old
from backend.app.features.human_pose.services.ensemble import (COCO_KEYPOINT_SIGMAS,compute_disagreement,compute_oks_suspicion,compute_ranking_score,compute_reference_point,compute_reliability,compute_rho)
from experiments.benchmark_pose.bench_pose_mapping import COCO_TO_VF17,CROWDPOSE_TO_VF17
MODES={"K1":["yolo26s-pose"],"K2":["yolo26s-pose","rtmpose-m"],"K3":["yolo26s-pose","rtmpose-m","yolov8m-pose"]}
PAIRS={2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8,10:11,11:10,12:13,13:12,14:15,15:14,16:17,17:16}
def auc(y,s):
 y=np.asarray(y);s=np.asarray(s);p=y.sum();n=len(y)-p
 if not p or not n:return None
 o=np.argsort(s,kind="mergesort");r=np.empty(len(s));i=0
 while i<len(s):
  j=i+1
  while j<len(s) and s[o[j]]==s[o[i]]:j+=1
  r[o[i:j]]=(i+j+1)/2;i=j
 return float((r[y==1].sum()-p*(p+1)/2)/(p*n))
def ap(y,s):
 y=np.asarray(y);s=np.asarray(s);p=y.sum()
 if not p:return None
 yy=y[np.argsort(-s,kind="mergesort")];return float(np.sum(np.cumsum(yy)/(np.arange(len(y))+1)*yy)/p)
def extract(items,preds,mapping):
 out=[]
 for im in items:
  iid=im["image_id"]
  for ai,ann in enumerate(im["annotations"]):
   raw=ann["keypoints"];gt={};vis={}
   for ci in mapping:
    x,y,v=raw[3*ci:3*ci+3]
    if v>0:gt[mapping[ci]]=(float(x),float(y));vis[mapping[ci]]=int(v)
   if not gt:continue
   matches={m:old.match_gt_to_predicted_person(gt,preds[m].get(iid,[])) for m in preds}
   scale=math.sqrt(max(10,ann["bbox"][2])*max(10,ann["bbox"][3]))
   for pid,xy in gt.items():
    models={}
    for m,person in matches.items():
     if person:
      q=person.get("keypoints",{}).get(pid) or person.get("keypoints",{}).get(str(pid))
      if q and q.get("x") is not None:models[m]={"x":float(q["x"]),"y":float(q["y"]),"conf":float(q.get("conf",1))}
    out.append({"image_id":iid,"ann_idx":ai,"vf_id":pid,"gt":xy,"visibility":vis[pid],"scale":scale,"models":models,"gt_all":gt})
 return out
def reference(inst,models,weights):
 obs=[(inst["models"][m]["x"],inst["models"][m]["y"],inst["models"][m]["conf"],weights[m]) for m in models if m in inst["models"]]
 if not obs:return None
 p=compute_reference_point(obs);d=compute_disagreement(obs,p,inst["scale"]);c=sum(x[2]*x[3] for x in obs)/sum(x[3] for x in obs);r=compute_reliability(c,d,tau=.08,rho=compute_rho(len(obs),len(models)));return p,d,r
def score(inst,human,mode,weights,swap_ratio=.6,swap_min=.15,state="visible"):
 ref=reference(inst,MODES[mode],weights)
 if not ref:return 0.0
 p,d,r=ref;dist=math.hypot(human[0]-p[0],human[1]-p[1]);e=compute_oks_suspicion(dist,inst["scale"],d,COCO_KEYPOINT_SIGMAS.get(inst["vf_id"],.07),lam=.5);s=compute_ranking_score(e,r,alpha=.5)
 other=inst["gt_all"].get(PAIRS.get(inst["vf_id"],-1))
 if other:
  orig=dist+math.hypot(other[0]-other[0],other[1]-other[1]);sw=math.hypot(human[0]-other[0],human[1]-other[1])+math.hypot(other[0]-p[0],other[1]-p[1])
  if sw<orig*swap_ratio and orig>inst["scale"]*swap_min:s=max(s,.88)
 if state=="occluded":s*=.5
 return s
def rows(instances,mode,weights,seed,swap_ratio=.6,swap_min=.15):
 rng=random.Random(seed);out=[]
 for x in instances:
  gx,gy=x["gt"];sc=x["scale"]
  if rng.random()>=.05:typ,label,h,state="clean",0,(gx+rng.gauss(0,.012*sc),gy+rng.gauss(0,.012*sc)),"visible"
  else:
   choices=["small_shift","large_shift","wrong_occluded_state"]
   if x["vf_id"] in PAIRS and PAIRS[x["vf_id"]] in x["gt_all"]:choices.append("left_right_swap")
   typ=rng.choice(choices);label=1;state="visible"
   if typ=="small_shift":h=(gx+rng.uniform(.03,.06)*sc,gy)
   elif typ=="large_shift":h=(gx+rng.uniform(.10,.20)*sc,gy)
   elif typ=="left_right_swap":h=x["gt_all"][PAIRS[x["vf_id"]]]
   else:h=(gx,gy);state="occluded"
  out.append({"image_id":x["image_id"],"ann_idx":x["ann_idx"],"vf_id":x["vf_id"],"visibility":x["visibility"],"type":typ,"label":label,"score":score(x,h,mode,weights,swap_ratio,swap_min,state)})
 return out
def metric(r,threshold):
 y=np.array([x["label"] for x in r]);s=np.array([x["score"] for x in r]);p=s>=threshold;tp=int(np.sum(p&(y==1)));fp=int(np.sum(p&(y==0)));fn=int(np.sum((~p)&(y==1)));tn=int(np.sum((~p)&(y==0)))
 return {"n":len(r),"auroc":auc(y,s),"auprc":ap(y,s),"precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/(tp+fn) if tp+fn else 0,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,"fpr":fp/(fp+tn) if fp+tn else 0}
def boot(rows,n,seed):
 by=defaultdict(list)
 for r in rows:by[r["image_id"]].append(r)
 keys=list(by);rng=random.Random(seed);vals=defaultdict(list)
 for _ in range(n):
  z=[]
  for k in rng.choices(keys,k=len(keys)):z+=by[k]
  y=[x["label"] for x in z];s=[x["score"] for x in z];vals["auroc"].append(auc(y,s));vals["auprc"].append(ap(y,s))
 return {q:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for q,v in vals.items()}
def main():
 ap_=argparse.ArgumentParser();ap_.add_argument("--output",required=True);ap_.add_argument("--bootstrap",type=int,default=1000);ap_.add_argument("--workspace",default=str(ROOT/".bench_workspace/pose"));ap_.add_argument("--schema",choices=("coco","crowdpose"),default="coco");a=ap_.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);work=Path(a.workspace);old.CACHE_DIR=work/"cache";mapping=COCO_TO_VF17 if a.schema=="coco" else CROWDPOSE_TO_VF17
 man=json.loads((work/"data/split_manifest.json").read_text());models=list({m for v in MODES.values() for m in v});devp=old.load_cached_predictions(models,[x["image_id"] for x in man["dev"]]);testp=old.load_cached_predictions(models,[x["image_id"] for x in man["test"]]);dev=extract(man["dev"],devp,mapping);test=extract(man["test"],testp,mapping)
 weights={m:1/max(np.median([math.hypot(x["models"][m]["x"]-x["gt"][0],x["models"][m]["y"]-x["gt"][1])/x["scale"] for x in dev if m in x["models"]]),1e-4)**2 for m in models}
 result={"schema":a.schema,"dev_images":len(man["dev"]),"test_images":len(man["test"]),"dev_instances":len(dev),"test_instances":len(test),"weights":weights,"modes":{},"threshold_sweep":{},"swap_threshold_sweep":{}};raw_rows=[]
 for mode in MODES:
  swap_grid=[]
  for ratio in (.4,.5,.6,.7,.8):
   for minimum in (.1,.15,.2):swap_grid.append({"ratio":ratio,"min_scale":minimum,**metric(rows(dev,mode,weights,42,ratio,minimum),.2)})
  swap_choice=max(swap_grid,key=lambda x:(x["f1"],-x["fpr"],-abs(x["ratio"]-.6)-abs(x["min_scale"]-.15)))
  result["swap_threshold_sweep"][mode]={"dev":swap_grid,"selected":{"ratio":swap_choice["ratio"],"min_scale":swap_choice["min_scale"]}}
  dr=rows(dev,mode,weights,42,swap_choice["ratio"],swap_choice["min_scale"]);tr=rows(test,mode,weights,123,swap_choice["ratio"],swap_choice["min_scale"]);grid=[]
  for th in (.05,.1,.15,.2,.25,.3,.4):grid.append({"threshold":th,**metric(dr,th)})
  chosen=max(grid,key=lambda x:(x["f1"],-x["fpr"]));result["threshold_sweep"][mode]={"dev":grid,"selected":chosen["threshold"]};allm=metric(tr,chosen["threshold"]);allm["current_threshold_0_20"]=metric(tr,.2);allm["ci95_auroc_auprc"]=boot(tr,a.bootstrap,100+len(mode));bytype={};raw_rows.extend({"mode":mode,**x} for x in tr)
  for typ in ("small_shift","large_shift","left_right_swap","wrong_occluded_state"):
   sub=[x for x in tr if x["type"] in ("clean",typ)];bytype[typ]=metric(sub,chosen["threshold"])
  result["modes"][mode]={"selected_threshold":chosen["threshold"],"test":allm,"by_error_type":bytype}
 (out/"pose_qa_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 with (out/"pose_qa_test_rows.csv").open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=["mode","image_id","ann_idx","vf_id","visibility","type","label","score"]);writer.writeheader();writer.writerows(raw_rows)
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
