"""Empirical audit of the two documented VF50 config/runtime mismatches."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.app.features.face_landmark_vf50 import spec
from backend.app.features.face_landmark_vf50.rules.validation import check_all_rules
from experiments.benchmark_face.bench_phase2 import kdict

def main(output):
 d=json.loads((ROOT/".bench_workspace/prepared/wflw_landmark_qa_v1.json").read_text(encoding="utf-8"));cases=[]
 for im in d["images"]:
  if im["split"]=="test":cases += [kdict(a) for a in im["annotations"]]
  if len(cases)>=100:break
 cases=cases[:100];r10=0
 for base in cases:
  h=copy.deepcopy(base)
  for p in range(30,37):h[p].update(state="outside",x=None,y=None)
  r10 += "R10" in {v.code for v in check_all_rules(h)}
 original=spec.get_rules_spec; counts=[]
 for multiplier in (.1,100.0):
  rules=copy.deepcopy(spec.load_vf50_spec()["rules"]);rules["r09_inner_lip"]["contour_tolerance_multiplier"]=multiplier;spec.get_rules_spec=lambda rules=rules:rules
  counts.append(sum("R09" in {v.code for v in check_all_rules(h)} for h in cases))
 spec.get_rules_spec=original
 result={"n":len(cases),"r10_partial_outer_lip_detected":r10,"r10_coverage":r10/len(cases),"r09_contour_multiplier_values":[.1,100.0],"r09_fired_counts":counts,"r09_behavior_changed":counts[0]!=counts[1]}
 Path(output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main(sys.argv[1])
