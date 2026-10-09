import type { SkeletonConfig } from './types';

export const SKELETONS: SkeletonConfig[] = [
  {
    name: 'longmaytrai',
    label: 'Lông mày trái',
    pointRange: [0, 4],
    pointCount: 5,
    color: '#FF0000', // Đỏ (Red) chuẩn CVAT #FF0000
    closed: false,
    cvatId: 116,
    description: 'Bờ trên lông mày trái: 0 đuôi ngoài -> 4 đầu trong',
  },
  {
    name: 'longmayphai',
    label: 'Lông mày phải',
    pointRange: [5, 9],
    pointCount: 5,
    color: '#3399FF', // Xanh lam nhạt (Sky Blue) chuẩn CVAT #3399FF
    closed: false,
    cvatId: 122,
    description: 'Bờ trên lông mày phải: 5 đầu trong -> 9 đuôi ngoài',
  },
  {
    name: 'songmui',
    label: 'Sống mũi',
    pointRange: [10, 13],
    pointCount: 4,
    color: '#FFFF00', // Vàng tươi (Yellow) chuẩn CVAT #FFFF00
    closed: false,
    cvatId: 128,
    description: 'Sống mũi: 10 đỉnh sống mũi -> 13 chân sống mũi (trên cánh mũi)',
  },
  {
    name: 'mattrai',
    label: 'Mắt trái',
    pointRange: [14, 21],
    pointCount: 8,
    color: '#FF00FF', // Hồng sen / Magenta chuẩn CVAT #FF00FF
    closed: true,
    cvatId: 133,
    description: 'Mắt trái: 14 khoé ngoài -> 15-17 mí trên -> 18 khoé trong -> 19-21 mí dưới',
  },
  {
    name: 'matphai',
    label: 'Mắt phải',
    pointRange: [22, 29],
    pointCount: 8,
    color: '#00FFFF', // Xanh lơ / Cyan chuẩn CVAT #00FFFF
    closed: true,
    cvatId: 142,
    description: 'Mắt phải: 22 khoé trong -> 23-25 mí trên -> 26 khoé ngoài -> 27-29 mí dưới',
  },
  {
    name: 'moingoai',
    label: 'Viền môi ngoài',
    pointRange: [30, 41],
    pointCount: 12,
    color: '#00FF00', // Xanh lá chuối / Bright Green chuẩn CVAT #00FF00
    closed: true,
    cvatId: 151,
    description: 'Viền môi ngoài: 30 khoé trái -> 31-35 môi trên -> 36 khoé phải -> 37-41 môi dưới',
  },
  {
    name: 'moitrong',
    label: 'Viền môi trong',
    pointRange: [42, 49],
    pointCount: 8,
    color: '#FF9900', // Cam tươi / Orange chuẩn CVAT #FF9900
    closed: true,
    cvatId: 164,
    description: 'Viền môi trong: 42 khoé trái -> 43-45 mép trên -> 46 khoé phải -> 47-49 mép dưới',
  },
];

export function getPointColor(pointId: number): string {
  if (pointId >= 0 && pointId <= 4) return '#FF0000'; // longmaytrai
  if (pointId >= 5 && pointId <= 9) return '#3399FF'; // longmayphai
  if (pointId >= 10 && pointId <= 13) return '#FFFF00'; // songmui
  if (pointId >= 14 && pointId <= 21) return '#FF00FF'; // mattrai
  if (pointId >= 22 && pointId <= 29) return '#00FFFF'; // matphai
  if (pointId >= 30 && pointId <= 41) return '#00FF00'; // moingoai
  if (pointId >= 42 && pointId <= 49) return '#FF9900'; // moitrong
  return '#10B981';
}

export function getSkeletonByPointId(pointId: number): SkeletonConfig | undefined {
  return SKELETONS.find((sk) => pointId >= sk.pointRange[0] && pointId <= sk.pointRange[1]);
}

export const ANCHOR_POINTS = new Set([0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36]);

export const RULE_NAMES: Record<string, string> = {
  R01: 'Thiếu điểm hoặc số lượng sai',
  R02: 'Tọa độ ngoài ảnh (1280x720)',
  R03: 'Mí trên thấp hơn mí dưới (Lật mí)',
  R04: 'Đường viền tự cắt chéo (Hình chữ X)',
  R05: 'Khoảng cách sống mũi không đều (> 1.45x)',
  R06: 'Nhầm lẫn Trái / Phải (Viewpoint)',
  R07: 'Môi trong tràn ra ngoài môi ngoài',
  R08: 'Trạng thái điểm sai (Gán đè kính)',
  R09: 'Tọa độ nhảy vọt giữa 2 frame (> 15px)',
  R10: 'Góc nghiêng mắt bất thường (> 45 deg)',
  R11: 'Sai lệch điểm neo (> 3% IOD)',
  R12: 'Sai lệch đường viền (> 5% IOD)',
  R13: 'Lệch trung bình toàn ảnh NME (> 0.035)',
  R14: 'Mắt/Môi nhắm nhưng điểm biến mất',
};

export const POINT_ANATOMY_NAMES: Record<number, string> = {
  0: 'Đuôi ngoài lông mày trái (Neo)',
  1: 'Bờ ngoài lông mày trái',
  2: 'Đỉnh vòm lông mày trái',
  3: 'Bờ trong lông mày trái',
  4: 'Đầu trong lông mày trái (Neo)',
  5: 'Đầu trong lông mày phải (Neo)',
  6: 'Bờ trong lông mày phải',
  7: 'Đỉnh vòm lông mày phải',
  8: 'Bờ ngoài lông mày phải',
  9: 'Đuôi ngoài lông mày phải (Neo)',
  10: 'Gốc sống mũi / Giữa hai mắt (Neo)',
  11: 'Đoạn sống mũi trên',
  12: 'Đoạn sống mũi dưới',
  13: 'Chân sống mũi ngang cánh mũi (Neo)',
  14: 'Khóe ngoài mắt trái (Neo)',
  15: 'Bờ mí trên ngoài mắt trái',
  16: 'Đỉnh mí trên mắt trái',
  17: 'Bờ mí trên trong mắt trái',
  18: 'Khóe trong mắt trái (Neo)',
  19: 'Bờ mí dưới trong mắt trái',
  20: 'Đáy mí dưới mắt trái',
  21: 'Bờ mí dưới ngoài mắt trái',
  22: 'Khóe trong mắt phải (Neo)',
  23: 'Bờ mí trên trong mắt phải',
  24: 'Đỉnh mí trên mắt phải',
  25: 'Bờ mí trên ngoài mắt phải',
  26: 'Khóe ngoài mắt phải (Neo)',
  27: 'Bờ mí dưới ngoài mắt phải',
  28: 'Đáy mí dưới mắt phải',
  29: 'Bờ mí dưới trong mắt phải',
  30: 'Khóe môi trái (Neo)',
  31: 'Bờ môi trên trái',
  32: 'Đỉnh cánh môi trên trái',
  33: 'Nhân trung / Đáy tim môi trên',
  34: 'Đỉnh cánh môi trên phải',
  35: 'Bờ môi trên phải',
  36: 'Khóe môi phải (Neo)',
  37: 'Bờ môi dưới phải',
  38: 'Đáy viền môi dưới phải',
  39: 'Đáy viền môi dưới giữa',
  40: 'Đáy viền môi dưới trái',
  41: 'Bờ môi dưới trái',
  42: 'Khóe môi trong trái',
  43: 'Mép môi trong trên trái',
  44: 'Giữa mép môi trong trên',
  45: 'Mép môi trong trên phải',
  46: 'Khóe môi trong phải',
  47: 'Mép môi trong dưới phải',
  48: 'Giữa mép môi trong dưới',
  49: 'Mép môi trong dưới trái',
};
