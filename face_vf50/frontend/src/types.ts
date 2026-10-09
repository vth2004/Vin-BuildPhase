export interface Keypoint {
  id: number;
  x: number | null;
  y: number | null;
  state: 'visible' | 'occluded' | 'outside';
  skeleton?: string;
}

export interface RuleViolation {
  code: string;
  rule_name: string;
  level: 'ERROR' | 'WARNING';
  severity: 'critical' | 'major' | 'minor';
  message: string;
  points: number[];
  gl_section: string;
}

export interface FrameListItem {
  id: number;
  session_id: string;
  frame_index: number;
  image_filename: string;
  iod: number;
  nme: number;
  severity_score: number;
  error_count: number;
  status: 'pending' | 'reviewed' | 'fixed';
  rule_violations: RuleViolation[];
  case_type?: string;
  ai_reliability?: number;
  case_label?: string;
  case_description?: string;
}

export interface FrameDetail extends FrameListItem {
  image_path: string;
  has_human_labels: number;
  has_model_prediction: number;
  human_keypoints: Record<string, Keypoint>;
  model_keypoints: Record<string, Keypoint>;
}

export interface SessionInfo {
  session_id: string;
  name: string;
  created_at: string;
  total_frames: number;
  total_issues: number;
  avg_nme: number;
}

export interface FramesResponse {
  total: number;
  page: number;
  page_size: number;
  items: FrameListItem[];
}

export interface SkeletonConfig {
  name: string;
  label: string;
  pointRange: [number, number];
  pointCount: number;
  color: string;
  closed: boolean;
  description: string;
  cvatId?: number;
}
