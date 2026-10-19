export type { Session } from '../../shared/types'
export type { Dataset } from '../../shared/types'
export type FaceRun = {
  id: string
  dataset_id: string
  dataset_name: string
  status: string
  progress: number
  mode: string
  error_message?: string
  created_at: number
  temporal_enabled?: boolean
  temporal_window_size?: number
  input_mode?: string
}
export type Point = {
  id: number
  name?: string
  x: number | null
  y: number | null
  state: 'visible' | 'occluded' | 'outside'
  source?: 'human' | 'model'
}
export type Violation = {
  code: string
  rule_name: string
  level: 'ERROR' | 'WARNING'
  message: string
  points: number[]
  gl_section: string
}
export type FaceResultSummary = {
  id: string
  frame_index: number
  face_index: number
  file_name: string
  width: number
  height: number
  iod: number
  nme: number | null
  severity: number
  error_count: number
  violation_count?: number
  result_kind?: string
  iod_source?: string
  detector_confidence?: number | null
  status: string
  case_type: string
  ai_reliability: number
  case_label: string
  violations: Violation[]
  e_max?: number | null
  e_p90?: number | null
  exceed_count?: number
  has_outlier_point?: boolean
  evaluated_points_count?: number
}
export type FaceResultDetail = FaceResultSummary & {
  human_keypoints: Record<string, Point>
  model_keypoints: Record<string, Point>
  reviewed_keypoints: Record<string, Point>
  point_sources: Record<string, 'human' | 'model'>
  quality: {
    iod?: number
    nme?: number | null
    nme_pct?: number | null
    passed_nme?: boolean | null
    e_max?: number | null
    e_p90?: number | null
    exceed_count?: number
    has_outlier_point?: boolean
    evaluated_points_count?: number
    point_evaluations?: Record<string, { exceeded: boolean; dist_px: number; norm_error_pct: number; severity?: string; is_anchor?: boolean }>
    case_description?: string
  }
}
