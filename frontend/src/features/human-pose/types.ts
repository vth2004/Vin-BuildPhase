export type Warning = {
  id: string
  run_id: string
  image_name: string
  keypoint: string
  warning_type: string
  suspicion: number
  human_x: number
  human_y: number
  suggested_x: number
  suggested_y: number
  review: string | null
  reliability?: number
  delta?: number
}

export type KeypointDetail = {
  id: number
  name?: string
  x: number | null
  y: number | null
  state: string
}

export type ImageDetails = {
  keypoints: KeypointDetail[]
  edges: [number, number][]
  width: number
  height: number
}
