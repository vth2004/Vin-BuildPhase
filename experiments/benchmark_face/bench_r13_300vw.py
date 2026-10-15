"""Measure production R13 false positives on real consecutive 300VW GT.

Reads .pts annotations directly from the user-provided ZIP; no dataset files
are extracted or modified. The fixed subset covers all official test groups.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re, sys, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend.app.features.face_landmark_vf50.rules.geometry import compute_iod
from backend.app.features.face_landmark_vf50.rules.validation import check_all_rules
from experiments.benchmark_face.bench_mapping_core import load_mapping,map_points

SUBSET={"train":["001"],"test_category_1":["114","124","125"],
        "test_category_2":["203","208","211"],"test_category_3":["410","411","516"]}

def parse_pts(raw):
 text=raw.decode("ascii","replace"); body=text[text.index("{")+1:text.rindex("}")]
 pts=np.asarray([[float(v) for v in line.split()] for line in body.splitlines() if line.strip()],float)
 if pts.shape!=(68,2): raise ValueError(f"expected 68x2, got {pts.shape}")
 return pts
def kd(points):return {i:{"id":i,"x":float(x),"y":float(y),"state":"visible"} for i,(x,y) in enumerate(points)}
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--zip",required=True);ap.add_argument("--output",required=True)
 ap.add_argument("--frames-per-video",type=int,default=300);ap.add_argument("--bootstrap",type=int,default=1000);ap.add_argument("--seed",type=int,default=42);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=load_mapping(ROOT/"experiments/benchmark_face/configs/ibug68_to_vf50.yaml")
 rows=[];seq_summ=[]
 with zipfile.ZipFile(a.zip) as z:
  names=set(z.namelist())
  for category,seqs in SUBSET.items():
   for seq in seqs:
    prefix=f"300VW_Dataset_2015_12_14/{seq}/annot/"; fs=sorted(n for n in names if n.startswith(prefix) and n.endswith(".pts"))[:a.frames_per_video]
    prev=None;fp=0;pairs=0
    for name in fs:
     cur=kd(map_points(parse_pts(z.read(name)),cfg)); fired=[]
     if prev is not None:
      fired=sorted({v.code for v in check_all_rules(cur,prev_keypoints_by_id=prev,iod=max(compute_iod(cur),10.0))})
      pairs+=1;fp+=int("R13" in fired)
     rows.append({"category":category,"sequence":seq,"frame":Path(name).stem,"pair_evaluated":int(prev is not None),"r13_fired":int("R13" in fired),"iod":compute_iod(cur)})
     prev=cur
    seq_summ.append({"category":category,"sequence":seq,"frames":len(fs),"pairs":pairs,"false_positive":fp,"fpr":fp/pairs if pairs else None})
 with (out/"r13_300vw_frames.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 total=sum(x["pairs"] for x in seq_summ);fps=sum(x["false_positive"] for x in seq_summ)
 rng=random.Random(a.seed);boots=[]
 for _ in range(a.bootstrap):
  s=rng.choices(seq_summ,k=len(seq_summ)); boots.append(sum(x["false_positive"] for x in s)/sum(x["pairs"] for x in s))
 summary={"seed":a.seed,"subset":SUBSET,"frames_per_video":a.frames_per_video,"videos":len(seq_summ),"frames":len(rows),"pairs":total,"false_positive":fps,"fpr":fps/total,"bootstrap_unit":"video","bootstrap":a.bootstrap,"fpr_ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"per_sequence":seq_summ,
 "source_zip":str(Path(a.zip)),"source_zip_sha256":sha256_file(a.zip),"mapping":"experiments/benchmark_face/configs/ibug68_to_vf50.yaml"}
 (out/"r13_300vw_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
