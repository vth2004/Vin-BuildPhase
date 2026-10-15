"""Create reproducible Phase-2 tables from raw CSV outputs."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np

def roc(y,s):
 y=np.asarray(y);s=np.asarray(s);p=y.sum();n=len(y)-p
 if not p or not n:return None
 o=np.argsort(s,kind="mergesort");r=np.empty(len(s),float);i=0
 while i<len(s):
  j=i+1
  while j<len(s) and s[o[j]]==s[o[i]]:j+=1
  r[o[i:j]]=(i+j+1)/2;i=j
 return float((r[y==1].sum()-p*(p+1)/2)/(p*n))
def pr(y,s):
 y=np.asarray(y);s=np.asarray(s);p=y.sum()
 if not p:return None
 yy=y[np.argsort(-s,kind="mergesort")];return float(np.sum(np.cumsum(yy)/(np.arange(len(y))+1)*yy)/p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("result");a=ap.parse_args();d=Path(a.result)
 scenarios=list(csv.DictReader((d/"scenario_results.csv").open(encoding="utf-8")))
 by=defaultdict(lambda:[0,0])
 for r in scenarios:
  fired=set(json.loads(r["fired"])); exp=r["expected_rule"]
  if exp: by[(r["injection"],r["level_iod"])][0]+=1;by[(r["injection"],r["level_iod"])][1]+=exp in fired
 detection=[{"injection":k[0],"level_iod":float(k[1]),"n":v[0],"detected":v[1],"recall":v[1]/v[0]} for k,v in sorted(by.items())]
 clean_scenarios=[r for r in scenarios if r["injection"]=="clean"]
 clean_fpr=[{"rule":rule,"n":len(clean_scenarios),"false_positive":sum(rule in set(json.loads(r["fired"])) for r in clean_scenarios),
             "fpr":sum(rule in set(json.loads(r["fired"])) for r in clean_scenarios)/len(clean_scenarios)} for rule in [f"R{i:02d}" for i in range(1,15)]]
 clean={}; groups=defaultdict(list); by_case=defaultdict(list)
 with (d/"point_scores.csv").open(encoding="utf-8") as f:
  for r in csv.DictReader(f):
   item=(int(r["is_error"]),float(r["norm_error"]),float(r["suspicion"]),{"NORMAL":0,"WARNING":1,"CRITICAL":2}[r["severity"]])
   key=(r["case_id"],r["point_id"])
   if r["injection"]=="clean":clean[key]=item
   else:
    groups[(r["injection"],r["level_iod"])].append(item)
    by_case[(r["injection"],r["level_iod"],r["case_id"])].append(item)
 curves=[]
 cleanvals=list(clean.values())
 for k,v in sorted(groups.items()):
  z=cleanvals+v;y=[x[0] for x in z]
  row={"injection":k[0],"level_iod":float(k[1]),"positive_points":sum(y),"total_points":len(z)}
  for name,idx in (("norm_error",1),("suspicion",2),("severity",3)):
   row[name+"_auroc"]=roc(y,[x[idx] for x in z]);row[name+"_auprc"]=pr(y,[x[idx] for x in z])
  # Macro-by-face bootstrap: each face contributes one AUC/AP computed from
  # its clean 50 points plus the corresponding injected scenario.
  case_stats=defaultdict(list)
  for (inj,lev,cid),items in by_case.items():
   if (inj,lev)!=k:continue
   z=[clean[(cid,str(pid))] for pid in range(50)]+items;y0=[x[0] for x in z]
   for name,idx in (("norm_error",1),("suspicion",2),("severity",3)):
    case_stats[name+"_auroc"].append(roc(y0,[x[idx] for x in z]));case_stats[name+"_auprc"].append(pr(y0,[x[idx] for x in z]))
  rng=np.random.default_rng(42)
  for name,values in case_stats.items():
   arr=np.asarray(values,float);boots=np.mean(arr[rng.integers(0,len(arr),size=(1000,len(arr)))],axis=1)
   row[name+"_macro"]=float(arr.mean());row[name+"_macro_ci95"]=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]
  curves.append(row)
 (d/"injection_recall.json").write_text(json.dumps(detection,indent=2),encoding="utf-8")
 (d/"clean_control_fpr.json").write_text(json.dumps(clean_fpr,indent=2),encoding="utf-8")
 (d/"point_metrics_by_noise.json").write_text(json.dumps(curves,indent=2),encoding="utf-8")
 subprocess.run([str(Path("backend/.venv/Scripts/python.exe")),"-m","pip","freeze"],stdout=(d/"pip_freeze.txt").open("w",encoding="utf-8"),check=True)
 files=[p for p in d.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"]
 (d/"SHA256SUMS.txt").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(files))+"\n",encoding="utf-8")
 print(json.dumps({"injection_recall":detection,"point_metrics":curves},indent=2))
if __name__=="__main__":main()
