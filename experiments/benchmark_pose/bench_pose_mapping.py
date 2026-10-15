"""Explicit public Pose schemas to VF17 (right IDs even, left IDs odd)."""
COCO_TO_VF17={0:1,1:3,2:2,3:5,4:4,5:7,6:6,7:9,8:8,9:11,10:10,11:13,12:12,13:15,14:14,15:17,16:16}
# CrowdPose has only 14 joints. top_head/neck have no semantic VF17 match;
# VF17 face IDs 1..5 are unavailable and must remain missing, never invented.
CROWDPOSE_TO_VF17={0:7,1:6,2:9,3:8,4:11,5:10,6:13,7:12,8:15,9:14,10:17,11:16}
def map_points(points,mapping):return {vf_id:points[src_id] for src_id,vf_id in mapping.items()}
