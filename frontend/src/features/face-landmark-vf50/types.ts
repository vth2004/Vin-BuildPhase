export type Session = { projectId: string; recoveryKey: string; projectName: string }
export type Dataset = { id: string; name: string; status: string; image_count: number; schema_id: string }
export type FaceRun = {
  id: string
  dataset_id: string
  dataset_name: string
  status: string
  progress: number
  mode: string
  error_message?: string
  created_at: number
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
  nme: number
  severity: number
  error_count: number
  status: string
  case_type: string
  ai_reliability: number
  case_label: string
  violations: Violation[]
}
export type FaceResultDetail = FaceResultSummary & {
  human_keypoints: Record<string, Point>
  model_keypoints: Record<string, Point>
  reviewed_keypoints: Record<string, Point>
  point_sources: Record<string, 'human' | 'model'>
  quality: {
    point_evaluations?: Record<string, { exceeded: boolean; dist_px: number; norm_error_pct: number }>
    case_description?: string
  }
}
