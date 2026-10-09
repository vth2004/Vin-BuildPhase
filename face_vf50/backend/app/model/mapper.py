"""VinFast VF-50 Landmark Mapper.

Maps Google MediaPipe Face Mesh (478 dense landmarks) to VinFast VF-50
specification (50 landmarks across 7 skeletons) via arc-length polyline
resampling to guarantee smooth and guideline-compliant landmark distribution.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def resample_polyline(
    points: np.ndarray,
    fractions: List[float]
) -> np.ndarray:
    """Resample a 2D polyline along its cumulative arc-length at given fractions.
    
    Args:
        points: (N, 2) or (N, 3) array of coordinate points.
        fractions: List of floats in [0.0, 1.0].
        
    Returns:
        (len(fractions), D) array of resampled points.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= 1:
        return np.repeat(pts[:1], len(fractions), axis=0)
    
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs[:, :2], axis=1)  # arc length in 2D
    cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_length = cum_lengths[-1]
    
    if total_length < 1e-6:
        return np.repeat(pts[:1], len(fractions), axis=0)
    
    resampled = []
    for f in fractions:
        f_clamped = max(0.0, min(1.0, float(f)))
        target_dist = f_clamped * total_length
        idx = int(np.searchsorted(cum_lengths, target_dist, side="right") - 1)
        idx = max(0, min(idx, len(cum_lengths) - 2))
        
        seg_dist = target_dist - cum_lengths[idx]
        seg_len = seg_lengths[idx]
        t = (seg_dist / seg_len) if seg_len > 1e-6 else 0.0
        interp_pt = pts[idx] + t * (pts[idx + 1] - pts[idx])
        resampled.append(interp_pt)
        
    return np.array(resampled, dtype=np.float32)


# Canonical MediaPipe Face Mesh indices for contours
MP_LEFT_EYEBROW = [70, 63, 105, 66, 107]
MP_RIGHT_EYEBROW = [336, 296, 334, 293, 300]
MP_NOSE_BRIDGE = [168, 6, 197, 195]

MP_LEFT_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
MP_LEFT_EYE_LOWER = [133, 155, 154, 153, 145, 144, 163, 7, 33]

MP_RIGHT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]
MP_RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]

MP_OUTER_LIP_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
MP_OUTER_LIP_LOWER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

MP_INNER_LIP_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
MP_INNER_LIP_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


def map_mediapipe_to_vf50(
    mp_landmarks: np.ndarray,
    image_width: float = 1280.0,
    image_height: float = 720.0,
    normalized: bool = True
) -> np.ndarray:
    """Converts 478 MediaPipe landmarks to 50 VinFast landmarks in pixel coordinates.
    
    Args:
        mp_landmarks: Array of shape (478, 2) or (478, 3). If normalized, coords are in [0, 1].
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        normalized: If True, scales x by image_width and y by image_height.
        
    Returns:
        np.ndarray of shape (50, 2) containing (x, y) coordinates for all 50 VF landmarks.
    """
    pts = np.asarray(mp_landmarks, dtype=np.float32)
    if normalized:
        pts = pts.copy()
        pts[:, 0] *= image_width
        pts[:, 1] *= image_height

    vf50 = np.zeros((50, 2), dtype=np.float32)

    # 1. longmaytrai (0-4): 5 points
    # Outer tail to inner head
    eyebrow_l = pts[MP_LEFT_EYEBROW]
    vf50[0:5] = resample_polyline(eyebrow_l, [0.00, 0.15, 0.41, 0.73, 1.00])[:, :2]

    # 2. longmayphai (5-9): 5 points
    # Inner head to outer tail
    eyebrow_r = pts[MP_RIGHT_EYEBROW]
    vf50[5:10] = resample_polyline(eyebrow_r, [0.00, 0.26, 0.58, 0.85, 1.00])[:, :2]

    # 3. songmui (10-13): 4 points
    # Top glabella (10) to nose bridge foot (13)
    nose_bridge = pts[MP_NOSE_BRIDGE]
    vf50[10:14] = resample_polyline(nose_bridge, [0.00, 0.34, 0.67, 1.00])[:, :2]

    # 4. mattrai (14-21): 8 points
    # 14 (outer) -> 15-17 (upper) -> 18 (inner)
    eye_l_upper = pts[MP_LEFT_EYE_UPPER]
    res_l_upper = resample_polyline(eye_l_upper, [0.00, 0.25, 0.50, 0.75, 1.00])[:, :2]
    vf50[14:19] = res_l_upper  # points 14, 15, 16, 17, 18

    # 19-21 (lower) between 18 and 14
    eye_l_lower = pts[MP_LEFT_EYE_LOWER]
    res_l_lower = resample_polyline(eye_l_lower, [0.25, 0.50, 0.75])[:, :2]
    vf50[19:22] = res_l_lower  # points 19, 20, 21

    # 5. matphai (22-29): 8 points
    # 22 (inner) -> 23-25 (upper) -> 26 (outer)
    eye_r_upper = pts[MP_RIGHT_EYE_UPPER]
    res_r_upper = resample_polyline(eye_r_upper, [0.00, 0.25, 0.50, 0.75, 1.00])[:, :2]
    vf50[22:27] = res_r_upper  # points 22, 23, 24, 25, 26

    # 27-29 (lower) between 26 and 22
    eye_r_lower = pts[MP_RIGHT_EYE_LOWER]
    res_r_lower = resample_polyline(eye_r_lower, [0.25, 0.50, 0.75])[:, :2]
    vf50[27:30] = res_r_lower  # points 27, 28, 29

    # 6. moingoai (30-41): 12 points
    # 30 (left corner) -> 31-35 (upper) -> 36 (right corner)
    lip_o_upper = pts[MP_OUTER_LIP_UPPER]
    res_o_upper = resample_polyline(lip_o_upper, [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])[:, :2]
    vf50[30:37] = res_o_upper  # points 30..36

    # 37-41 (lower) between 36 and 30
    lip_o_lower = pts[MP_OUTER_LIP_LOWER]
    res_o_lower = resample_polyline(lip_o_lower, [1/6, 2/6, 3/6, 4/6, 5/6])[:, :2]
    vf50[37:42] = res_o_lower  # points 37..41

    # 7. moitrong (42-49): 8 points
    # 42 (left inner) -> 43-45 (upper) -> 46 (right inner)
    lip_i_upper = pts[MP_INNER_LIP_UPPER]
    res_i_upper = resample_polyline(lip_i_upper, [0.0, 0.25, 0.50, 0.75, 1.0])[:, :2]
    vf50[42:47] = res_i_upper  # points 42..46

    # 47-49 (lower) between 46 and 42
    lip_i_lower = pts[MP_INNER_LIP_LOWER]
    res_i_lower = resample_polyline(lip_i_lower, [0.25, 0.50, 0.75])[:, :2]
    vf50[47:50] = res_i_lower  # points 47..49

    return vf50
