"""DEV-only threshold selection followed by one locked TEST evaluation."""
from __future__ import annotations
import argparse, copy, json, random, sys, zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np, yaml

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.app.features.face_landmark_vf50 import spec
from backend.app.features.face_landmark_vf50.rules.geometry import compute_iod
from backend.app.features.face_landmark_vf50.rules.validation import check_all_rules
from experiments.benchmark_face.bench_phase2 import inject,kdict
from experiments.benchmark_face.bench_mapping_core import load_mapping,map_points
from experiments.benchmark_face.bench_r13_300vw import parse_pts,kd

CURRENT=spec.load_vf50_spec()
PARAMS={
 "r04_scale":("R04","r04_eyebrow_monotonic","scale_factor_multiplier",[2,3,4,5,6,8],5),
 "r06_scale":("R06","r06_eyelid_tilt","scale_factor_multiplier",[1,1.5,2,2.5,3,4],2),
 "r08_ratio":("R08","r08_nose_bridge","max_ratio",[1.3,1.4,1.55,1.7,2,2.5],1.55),
 "r09_point_px":("R09","r09_inner_lip","per_point_tolerance_px",[.5,1,1.5,2,3,4],2),
 "r12_iod_ratio":("R12","r12_coincident_points","iod_ratio_multiplier",[.004,.006,.008,.01,.012,.016],.012),
 "r14_drift":("R14","r14_occluded_drift","max_drift_iod_ratio",[.05,.08,.1,.12,.15,.2,.25],.15),
}
KINDS={"R04":"R04_brow_order","R06":"R06_lid_flip","R08":"R08_nose_uneven","R09":"R09_inner_outside","R12":"R12_duplicate","R14":"R14_occluded_far"}

def load_split(paths,split,seed,limit):
 out=[];rng=random.Random(seed)
 for p in paths:
  d=json.loads(Path(p).read_text(encoding="utf-8"));c=[]
  for im in d["images"]:
   if im["split"]!=split:continue
   for j,a in enumerate(im["annotations"]):c.append({"dataset":d["dataset"],"image":im["file_name"],"face":j,"width":im["width"],"height":im["height"],"points":kdict(a)})
  rng.shuffle(c);out+=c[:limit]
 return out
@contextmanager
def rules_override(section,key,value):
 original=spec.get_rules_spec; rules=copy.deepcopy(CURRENT["rules"]);rules[section][key]=value;spec.get_rules_spec=lambda:rules
 try:yield
 finally:spec.get_rules_spec=original
def score_cases(cases,rule,kind,section,key,value,seed):
 rng=random.Random(seed);tp=fp=tn=fn=0
 with rules_override(section,key,value):
  for c in cases:
   base=c["points"];iod=max(compute_iod(base),10)
   for positive in (False,True):
    h,_,prev=inject(base,kind if positive else "clean",.1,rng,iod)
    pred=rule in {v.code for v in check_all_rules(h,prev_keypoints_by_id=prev,model_keypoints_by_id=base,image_width=c["width"],image_height=c["height"],iod=iod)}
    tp+=positive and pred;fp+=(not positive) and pred;tn+=(not positive) and not pred;fn+=positive and not pred
 return met(tp,fp,tn,fn)
def met(tp,fp,tn,fn):
 return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/(tp+fn) if tp+fn else 0,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,"fpr":fp/(fp+tn) if fp+tn else 0}
def select(rows,current):return max(rows,key=lambda x:(x["dev"]["f1"],-x["dev"]["fpr"],-abs(x["value"]-current)))

def scoring_samples(cases,seed):
 rng=random.Random(seed);points=[];faces=[]
 for c in cases:
  base=c["points"];iod=max(compute_iod(base),10)
  for kind in ("gaussian_independent","correlated_contour","systematic_skeleton"):
   for level in (.01,.03,.05,.1,.2):
    h,changed,_=inject(base,kind,level,rng,iod);errs=[]
    for pid in range(50):
     e=np.hypot(h[pid]["x"]-base[pid]["x"],h[pid]["y"]-base[pid]["y"])/iod;errs.append(e)
     points.append((pid in changed,e,pid in set(CURRENT["anchor_points"]),level>=.1))
    faces.append((True,float(np.mean(errs))))
  faces.append((False,0.0))
 return points,faces
def binary(rows,pred):
 tp=fp=tn=fn=0
 for i,r in enumerate(rows):
  y=bool(r[0]);p=bool(pred(r));tp+=y and p;fp+=(not y) and p;tn+=(not y) and not p;fn+=y and not p
 return met(tp,fp,tn,fn)

def r13_sequences(zip_path,seqs,frames=300):
 cfg=load_mapping(ROOT/"experiments/benchmark_face/configs/ibug68_to_vf50.yaml");out=[]
 with zipfile.ZipFile(zip_path) as z:
  names=set(z.namelist())
  for seq in seqs:
   ns=sorted(n for n in names if f"/{seq}/annot/" in n and n.endswith(".pts"))[:frames]
   cur=[kd(map_points(parse_pts(z.read(n)),cfg)) for n in ns]
   out += list(zip(cur[:-1],cur[1:]))
 return out
def score_r13(pairs,value):
 rules=copy.deepcopy(CURRENT["rules"]);rules["r13_temporal_jump"]["scale_factor_multiplier"]=value
 tp=fp=tn=fn=0
 with rules_override("r13_temporal_jump","scale_factor_multiplier",value):
  for prev,cur in pairs:
   iod=max(compute_iod(cur),10)
   for positive in (False,True):
    h=copy.deepcopy(cur)
    if positive:h[18]["x"]+=.2*iod
    pred="R13" in {v.code for v in check_all_rules(h,prev_keypoints_by_id=prev,iod=iod)}
    tp+=positive and pred;fp+=(not positive) and pred;tn+=(not positive) and not pred;fn+=positive and not pred
 return met(tp,fp,tn,fn)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--zip-300vw",required=True);ap.add_argument("--limit",type=int,default=500);ap.add_argument("--seed",type=int,default=42);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 paths=[ROOT/".bench_workspace/prepared/wflw_landmark_qa_v1.json",ROOT/".bench_workspace/prepared/300w_landmark_qa_v1.json"]
 dev=load_split(paths,"dev",a.seed,a.limit);test=load_split(paths,"test",a.seed+1,a.limit)
 sweeps={};proposed={}
 for name,(rule,sec,key,grid,current) in PARAMS.items():
  rows=[]
  for v in grid:rows.append({"value":v,"dev":score_cases(dev,rule,KINDS[rule],sec,key,v,a.seed)})
  chosen=select(rows,current);chosen["test"]=score_cases(test,rule,KINDS[rule],sec,key,chosen["value"],a.seed+99);chosen["current_test"]=score_cases(test,rule,KINDS[rule],sec,key,current,a.seed+99)
  sweeps[name]=rows;proposed[name]=chosen
 # R13: fixed video split declared before scoring.
 dp=r13_sequences(a.zip_300vw,["001","114","203","410","411"]);tp=r13_sequences(a.zip_300vw,["124","125","208","211","516"])
 rr=[{"value":v,"dev":score_r13(dp,v)} for v in [10,15,20,25,30,40]];ch=select(rr,15);ch["test"]=score_r13(tp,ch["value"]);ch["current_test"]=score_r13(tp,15);sweeps["r13_scale"]=rr;proposed["r13_scale"]=ch
 # Scoring thresholds: coordinate-error label, independently selected on DEV.
 pd,fd=scoring_samples(dev,a.seed);pt,ft=scoring_samples(test,a.seed+99);score_out={}
 grids={"anchor_tolerance":([.015,.02,.025,.03,.04,.05],.03,lambda r,v:r[2] and r[1]>v,lambda r:r[2]),
        "contour_tolerance":([.02,.03,.04,.05,.06,.08],.05,lambda r,v:(not r[2]) and r[1]>v,lambda r:not r[2]),
        "nme_threshold":([.015,.02,.025,.03,.035,.04,.05],.035,lambda r,v:r[1]>v,lambda r:True),
        "severity_multiplier":([1.2,1.5,1.8,2,2.5,3],1.8,lambda r,v:r[1]>(.03 if r[2] else .05)*v,lambda r:True),
        "suspicion_q":([1.5,2,2.5,3,4],2.5,lambda r,v:min(1,r[1]/((.03 if r[2] else .05)*v))>=.5,lambda r:True)}
 for name,(grid,current,pred,flt) in grids.items():
  source=fd if name=="nme_threshold" else pd;target=ft if name=="nme_threshold" else pt
  rows=[{"value":v,"dev":binary([r for r in source if flt(r)],lambda r,v=v:pred(r,v))} for v in grid];ch=select(rows,current);ch["test"]=binary([r for r in target if flt(r)],lambda r:pred(r,ch["value"]));ch["current_test"]=binary([r for r in target if flt(r)],lambda r:pred(r,current));sweeps[name]=rows;score_out[name]=ch
 result={"seed":a.seed,"selection":"max DEV F1, then lower FPR, then closest to current","dev_faces":len(dev),"test_faces":len(test),"r13_dev_pairs":len(dp),"r13_test_pairs":len(tp),"rules":proposed,"scoring":score_out,"dev_sweeps":sweeps}
 (out/"phase3_threshold_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 proposal={"benchmark_only":True,"do_not_apply_automatically":True,
 "decision_note":"Only R12 is a provisional recommendation. Scoring candidates are rejected because clean GT equals model reference; R08/R09 gains need milder/real errors.",
 "recommended":{"rules":{"r12_coincident_points":{"iod_ratio_multiplier":proposed["r12_iod_ratio"]["value"]}}},
 "keep_current":{"scoring_tolerances":CURRENT["scoring_tolerances"],"rules":{"r04_eyebrow_monotonic":{"scale_factor_multiplier":5.0},"r06_eyelid_tilt":{"scale_factor_multiplier":2.0},"r08_nose_bridge":{"max_ratio":1.55},"r09_inner_lip":{"contour_tolerance_multiplier":2.0},"r13_temporal_jump":{"scale_factor_multiplier":15.0},"r14_occluded_drift":{"max_drift_iod_ratio":.15}}},
 "dev_selected_candidates_not_recommended":{"scoring_tolerances":{"anchor_point_iod_ratio":score_out["anchor_tolerance"]["value"],"contour_point_iod_ratio":score_out["contour_tolerance"]["value"],"nme_max_threshold":score_out["nme_threshold"]["value"],"severity_critical_multiplier":score_out["severity_multiplier"]["value"],"suspicion_normalizer_multiplier":score_out["suspicion_q"]["value"]},"rules":{"r08_nose_bridge":{"max_ratio":proposed["r08_ratio"]["value"]},"r09_inner_lip":{"per_point_tolerance_px":proposed["r09_point_px"]["value"],"warning":"runtime reads this undocumented key; YAML contour_tolerance_multiplier is ignored"}}}}
 (out/"proposed_vf50_thresholds.yaml").write_text(yaml.safe_dump(proposal,sort_keys=False),encoding="utf-8");print(json.dumps({"rules":proposed,"scoring":score_out},indent=2))
if __name__=="__main__":main()
