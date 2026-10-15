"""Phase 2: seeded fault injection against the production VF50 QA pipeline.

All mutations are made on copies of public GT.  The unmodified GT is used as
the model reference, which isolates QA rule/scoring behaviour from detector
error (detector coverage was measured separately in Phase 1).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.features.face_landmark_vf50.rules.geometry import compute_iod
from backend.app.features.face_landmark_vf50.rules.validation import check_all_rules
from backend.app.features.face_landmark_vf50.scoring.quality import evaluate_frame_quality

GROUPS = {"brow_l": range(0,5), "brow_r": range(5,10), "nose": range(10,14),
          "eye_l": range(14,22), "eye_r": range(22,30),
          "lip_outer": range(30,42), "lip_inner": range(42,50)}
RULES = [f"R{i:02d}" for i in range(1,15)]

def kdict(annotation):
    return {int(p["id"]): {"id": int(p["id"]), "x": p["x"], "y": p["y"], "state": p.get("state","visible")} for p in annotation["keypoints"]}

def load_cases(paths, seed, max_wflw, max_300w):
    rng=random.Random(seed); out=[]
    for path, limit in zip(paths, [max_wflw,max_300w]):
        d=json.loads(Path(path).read_text(encoding="utf-8")); candidates=[]
        for im in d["images"]:
            if im["split"] != "test": continue
            for j,a in enumerate(im["annotations"]):
                candidates.append({"dataset":d["dataset"],"image":im["file_name"],"face":j,
                  "width":im["width"],"height":im["height"],"attributes":a.get("attributes",{}),"points":kdict(a)})
        rng.shuffle(candidates); out.extend(candidates[:limit] if limit else candidates)
    return out

def setxy(h, pid, xy): h[pid]["x"],h[pid]["y"]=float(xy[0]),float(xy[1])
def swap_groups(h,a,b):
    for x,y in zip(a,b): h[x],h[y]=dict(h[y],id=x),dict(h[x],id=y)

def inject(base, kind, level, rng, iod):
    h=copy.deepcopy(base); changed=set(); prev=None
    dx=max(iod*level,2.0)
    if kind=="clean": return h,changed,prev
    if kind=="R01_missing": changed={rng.randrange(50)}; h.pop(next(iter(changed)))
    elif kind=="R02_state_unchanged": changed=set(range(50)) # all-visible is the fault condition
    elif kind=="R03_swap_labels": swap_groups(h,range(0,5),range(5,10)); swap_groups(h,range(14,22),range(22,30)); changed=set(range(30))-{10,11,12,13}
    elif kind=="R04_brow_order": h[1],h[3]=dict(h[3],id=1),dict(h[1],id=3); changed={1,3}
    elif kind=="R05_eye_corners": h[14],h[18]=dict(h[18],id=14),dict(h[14],id=18); changed={14,18}
    elif kind=="R06_lid_flip":
        y=max(h[15]["y"],h[21]["y"])+dx; h[15]["y"]=y; changed={15}
    elif kind=="R07_eye_cross": h[15],h[19]=dict(h[19],id=15),dict(h[15],id=19); changed={15,19}
    elif kind=="R08_nose_uneven":
        p10=np.array([h[10]["x"],h[10]["y"]]); p13=np.array([h[13]["x"],h[13]["y"]]); setxy(h,11,p10+.08*(p13-p10)); changed={11}
    elif kind=="R09_inner_outside": setxy(h,42,(h[30]["x"]-dx,h[30]["y"])); changed={42}
    elif kind=="R10_partial_hidden":
        changed={14,15,16,17,18};
        for p in changed: h[p]["state"]="outside"; h[p]["x"]=h[p]["y"]=None
    elif kind=="R11_out_of_bounds": setxy(h, rng.randrange(50),(-dx,-dx)); changed={p for p in h if h[p]["x"]<0}
    elif kind=="R12_duplicate": setxy(h,33,(h[32]["x"],h[32]["y"])); changed={33}
    elif kind=="R13_temporal_jump": prev=copy.deepcopy(base); h[18]["x"]+=max(dx,iod*.25); changed={18}
    elif kind=="R14_occluded_far": h[18]["state"]="occluded"; h[18]["x"]+=max(dx,iod*.25); changed={18}
    elif kind=="gaussian_independent":
        changed={rng.randrange(50)}
        for p in changed: h[p]["x"]+=rng.gauss(0,dx); h[p]["y"]+=rng.gauss(0,dx)
    elif kind=="correlated_contour":
        ids=list(GROUPS[rng.choice(["eye_l","eye_r","lip_outer","lip_inner"])]); changed=set(ids); ang=rng.random()*2*math.pi
        for p in ids: h[p]["x"]+=dx*math.cos(ang); h[p]["y"]+=dx*math.sin(ang)
    elif kind=="systematic_skeleton":
        ids=list(GROUPS[rng.choice(list(GROUPS))]); changed=set(ids)
        for p in ids: h[p]["x"]+=dx; h[p]["y"]+=dx*.35
    else: raise ValueError(kind)
    return h,changed,prev

def auc_roc(y,s):
    y=np.asarray(y); s=np.asarray(s); pos=y.sum(); neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(s,kind="mergesort"); ranks=np.empty(len(s),float); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2; i=j
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def auc_pr(y,s):
    y=np.asarray(y); s=np.asarray(s); pos=y.sum()
    if not pos:return None
    o=np.argsort(-s,kind="mergesort"); yy=y[o]; return float(np.sum(np.cumsum(yy)/(np.arange(len(y))+1)*yy)/pos)

def metrics(rows, group="all"):
    output=[]
    subsets={"all":rows}
    for ds in sorted({r["dataset"] for r in rows}): subsets[f"dataset:{ds}"]=[r for r in rows if r["dataset"]==ds]
    attrs=sorted({a for r in rows for a in r["attributes"] if r["attributes"].get(a)})
    for a in attrs: subsets[f"attribute:{a}"]=[r for r in rows if r["attributes"].get(a)]
    for name,sub in subsets.items():
      for rule in RULES:
        tp=fp=tn=fn=0
        for r in sub:
            yt=r["expected_rule"]==rule; yp=rule in r["fired"]
            tp+=yt and yp; fp+=(not yt) and yp; tn+=(not yt) and (not yp); fn+=yt and (not yp)
        output.append({"slice":name,"rule":rule,"n":len(sub),"tp":tp,"fp":fp,"tn":tn,"fn":fn,
          "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
          "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,"fpr":fp/(fp+tn) if fp+tn else None})
    return output

def bootstrap(rows, n, seed):
    # Pre-aggregate [TP,FP,TN,FN] per image/rule; cluster resampling then
    # becomes array summation instead of repeatedly rebuilding all slices.
    rng=np.random.default_rng(seed); by=defaultdict(list)
    for r in rows: by[(r["dataset"],r["image"])].append(r)
    clusters=[]
    for sub in by.values():
        a=np.zeros((len(RULES),4),dtype=np.int64)
        for ri,rule in enumerate(RULES):
            for r in sub:
                yt=r["expected_rule"]==rule; yp=rule in r["fired"]
                a[ri,0]+=yt and yp; a[ri,1]+=(not yt) and yp
                a[ri,2]+=(not yt) and (not yp); a[ri,3]+=yt and (not yp)
        clusters.append(a)
    clusters=np.stack(clusters); vals=defaultdict(list)
    for _ in range(n):
        z=clusters[rng.integers(0,len(clusters),size=len(clusters))].sum(axis=0)
        for ri,rule in enumerate(RULES):
            tp,fp,tn,fn=map(int,z[ri]); den=2*tp+fp+fn
            values={"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
                    "f1":2*tp/den if den else None,"fpr":fp/(fp+tn) if fp+tn else None}
            for field,value in values.items():
                if value is not None: vals[(rule,field)].append(value)
    return [{"rule":k[0],"metric":k[1],"low":float(np.percentile(v,2.5)),"high":float(np.percentile(v,97.5))} for k,v in vals.items()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--max-wflw",type=int,default=1000); ap.add_argument("--max-300w",type=int,default=300); ap.add_argument("--bootstrap",type=int,default=1000); args=ap.parse_args()
    root=ROOT; out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    paths=[root/".bench_workspace/prepared/wflw_landmark_qa_v1.json",root/".bench_workspace/prepared/300w_landmark_qa_v1.json"]
    cases=load_cases(paths,args.seed,args.max_wflw,args.max_300w); rng=random.Random(args.seed)
    structural=[("clean",None,.0)]+[(f"R{i:02d}_{n}",f"R{i:02d}",.1) for i,n in enumerate([
      "missing","state_unchanged","swap_labels","brow_order","eye_corners","lid_flip","eye_cross","nose_uneven","inner_outside","partial_hidden","out_of_bounds","duplicate","temporal_jump","occluded_far"],1)]
    coordinate=[(k,None,l) for k in ("gaussian_independent","correlated_contour","systematic_skeleton") for l in (.01,.03,.05,.10,.20)]
    rows=[]; points=[]
    for ci,c in enumerate(cases):
      base=c["points"]; iod=max(compute_iod(base),10.0)
      for kind,expected,level in structural+coordinate:
        h,changed,prev=inject(base,kind,level,rng,iod)
        vs=check_all_rules(h,prev_keypoints_by_id=prev,model_keypoints_by_id=base,image_width=c["width"],image_height=c["height"],iod=iod)
        q=evaluate_frame_quality(h,base,iod=iod); fired=sorted({v.code for v in vs})
        row={"case_id":ci,"dataset":c["dataset"],"image":c["image"],"face":c["face"],"attributes":c["attributes"],"injection":kind,"level_iod":level,"expected_rule":expected,"changed_points":sorted(changed),"fired":fired,"nme":q["nme"],"passed_nme":q["passed_nme"]}
        rows.append(row)
        if kind in {"clean","gaussian_independent","correlated_contour","systematic_skeleton"}:
          for pid,e in q["point_evaluations"].items(): points.append({"case_id":ci,"dataset":c["dataset"],"image":c["image"],"injection":kind,"level_iod":level,"point_id":pid,"is_error":int(pid in changed),"norm_error":e["norm_error"],"suspicion":e["suspicion"],"severity":e["severity"]})
    def write_csv(path,data):
      with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader()
        for x in data:
          z=dict(x)
          for k,v in z.items():
            if isinstance(v,(dict,list)):z[k]=json.dumps(v,ensure_ascii=False,separators=(",",":"))
          w.writerow(z)
    write_csv(out/"scenario_results.csv",rows); write_csv(out/"point_scores.csv",points)
    summary=metrics(rows); (out/"rule_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"rule_metrics_bootstrap.json").write_text(json.dumps(bootstrap(rows,args.bootstrap,args.seed+1),indent=2),encoding="utf-8")
    y=[p["is_error"] for p in points]; result={"seed":args.seed,"cases":len(cases),"scenarios":len(rows),"point_rows":len(points),"bootstrap":args.bootstrap,
      "point_detection":{"norm_error":{"auroc":auc_roc(y,[p["norm_error"] for p in points]),"auprc":auc_pr(y,[p["norm_error"] for p in points])},"suspicion":{"auroc":auc_roc(y,[p["suspicion"] for p in points]),"auprc":auc_pr(y,[p["suspicion"] for p in points])}},
      "limitations":["WFLW/300W mapped labels have no per-point occlusion state; R02 is unit/synthetic only and all-visible clean samples also fire R02.","GT-as-model isolates QA logic and is not an end-to-end detector benchmark; Phase 1 reports detector coverage."]}
    (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"run_config.json").write_text(json.dumps(vars(args)|{"inputs":[str(p) for p in paths]},indent=2),encoding="utf-8")
    (out/"commit_hash.txt").write_text(subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
