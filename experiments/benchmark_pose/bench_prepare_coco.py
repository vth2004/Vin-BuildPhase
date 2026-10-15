"""Prepare the existing deterministic 500-image COCO split in workspace."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/"experiments/model_check";sys.path.insert(0,str(OLD));sys.path.insert(0,str(ROOT))
import prepare_dataset as p
import json, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
WORK=ROOT/".bench_workspace/pose";p.DATA_DIR=WORK/"data";p.IMAGES_DIR=p.DATA_DIR/"images";p.ANNOTATIONS_DIR=p.DATA_DIR/"annotations";p.LOGS_DIR=WORK/"logs"
p.ANNOTATIONS_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip";p.COCO_IMAGE_BASE_URL="http://images.cocodataset.org/val2017"
def download_parallel():
 manifest=json.loads((p.DATA_DIR/"split_manifest.json").read_text(encoding="utf-8"));items=manifest["dev"]+manifest["test"]
 def one(item):
  dst=p.IMAGES_DIR/item["file_name"]
  if dst.exists() and dst.stat().st_size>1024:return "cached"
  tmp=dst.with_suffix(".tmp");req=urllib.request.Request(f"{p.COCO_IMAGE_BASE_URL}/{item['file_name']}",headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=60) as src,tmp.open("wb") as f:f.write(src.read())
  tmp.replace(dst);return "new"
 p.IMAGES_DIR.mkdir(parents=True,exist_ok=True);counts={"new":0,"cached":0}
 with ThreadPoolExecutor(max_workers=12) as ex:
  for i,f in enumerate(as_completed([ex.submit(one,x) for x in items]),1):counts[f.result()]+=1
 print({"images":len(items),**counts})
if __name__=="__main__":
 if (p.DATA_DIR/"split_manifest.json").exists():download_parallel()
 else:p.main()
