"""Create a deterministic 500-image CrowdPose DEV/TEST manifest."""
from __future__ import annotations
import json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/".bench_workspace/pose/crowdpose"
OUT=SRC/"benchmark"/"data"
def main():
 data=json.loads((SRC/"annotations/crowdpose_val.json").read_text(encoding="utf-8"));anns={}
 for a in data["annotations"]:anns.setdefault(a["image_id"],[]).append(a)
 items=[]
 for im in data["images"]:
  if im["id"] in anns and (SRC/"images"/im["file_name"]).is_file():items.append({"image_id":im["id"],"file_name":im["file_name"],"width":im["width"],"height":im["height"],"crowdIndex":im.get("crowdIndex"),"annotations":anns[im["id"]]})
 rng=random.Random(42);rng.shuffle(items);picked=items[:500];OUT.mkdir(parents=True,exist_ok=True)
 manifest={"dataset":"CrowdPose val","seed":42,"source_images":len(data["images"]),"dev":picked[:250],"test":picked[250:]}
 (OUT/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 print({"eligible":len(items),"dev":250,"test":250})
if __name__=="__main__":main()
