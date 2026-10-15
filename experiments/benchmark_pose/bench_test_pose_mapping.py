import unittest
from experiments.benchmark_pose.bench_pose_mapping import COCO_TO_VF17,CROWDPOSE_TO_VF17,map_points
class PoseMappingTests(unittest.TestCase):
 def test_vf_parity(self):
  for mapping in (COCO_TO_VF17,CROWDPOSE_TO_VF17):
   for src,vf in mapping.items():
    if vf>1:self.assertEqual(vf%2,0 if src in ({2,4,6,8,10,12,14,16} if mapping is COCO_TO_VF17 else {1,3,5,7,9,11}) else 1)
 def test_crowdpose_does_not_invent_face(self):self.assertTrue({1,2,3,4,5}.isdisjoint(CROWDPOSE_TO_VF17.values()))
 def test_no_swap_from_mapping(self):
  pts=[(0,0)]*14;pts[0]=(10,20);pts[1]=(90,20);m=map_points(pts,CROWDPOSE_TO_VF17);self.assertLess(m[7][0],m[6][0])
if __name__=="__main__":unittest.main()
