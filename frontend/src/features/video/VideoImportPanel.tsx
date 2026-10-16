import { FormEvent, useEffect, useRef, useState } from 'react'

import './video-import.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

type Session = { projectId: string; recoveryKey: string }
type ImportState = {
  dataset: { id: string; status: string; image_count: number }
  job: {
    id: string
    attempt: number
    status: string
    progress: number
    error_message: string | null
    max_frames: number
    max_decoded_bytes: number
  }
}

type Props = {
  session: Session
  schemaId: 'vf_humanpose17_v1' | 'vf_face_landmark50_v1'
  annotationAccept: string
  existingDatasetId?: string
  onReady: (datasetId: string) => void | Promise<void>
  onMessage?: (message: string) => void
}

export default function VideoImportPanel({ session, schemaId, annotationAccept, existingDatasetId, onReady, onMessage }: Props) {
  const [video, setVideo] = useState<File | null>(null)
  const [annotation, setAnnotation] = useState<File | null>(null)
  const [datasetId, setDatasetId] = useState('')
  const [state, setState] = useState<ImportState | null>(null)
  const [busy, setBusy] = useState(false)
  const notifiedReady = useRef('')

  const request = async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await fetch(`${API}${path}`, {
      ...init,
      headers: { Authorization: `Bearer ${session.recoveryKey}`, ...(init.headers || {}) },
    })
    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      const detail = typeof payload?.detail === 'string' ? payload.detail : payload?.detail?.message
      throw new Error(detail || `Request failed (${response.status})`)
    }
    return response.json()
  }

  const refresh = async (target = datasetId) => {
    if (!target) return
    const next = await request<ImportState>(`/projects/${session.projectId}/datasets/${target}/import`)
    setState(next)
    if (next.job.status === 'ready' && notifiedReady.current !== target) {
      notifiedReady.current = target
      onMessage?.(`Video import hoàn tất: ${next.dataset.image_count} frame.`)
      await onReady(target)
    }
  }

  useEffect(() => {
    if (!datasetId && existingDatasetId) {
      setDatasetId(existingDatasetId)
      refresh(existingDatasetId).catch(error => onMessage?.(error.message))
    }
  }, [existingDatasetId, datasetId])

  useEffect(() => {
    if (!datasetId || !state || !['pending', 'processing'].includes(state.job.status)) return
    const timer = window.setInterval(() => refresh(datasetId).catch(error => onMessage?.(error.message)), 1000)
    return () => window.clearInterval(timer)
  }, [datasetId, state?.job.status])

  const upload = async (event: FormEvent) => {
    event.preventDefault()
    if (!video) return
    setBusy(true)
    try {
      const body = new FormData()
      body.append('file', video)
      body.append('schema_id', schemaId)
      if (annotation) body.append('annotation_file', annotation)
      const created = await request<{ id: string }>(`/projects/${session.projectId}/datasets/video`, {
        method: 'POST', body,
      })
      notifiedReady.current = ''
      setDatasetId(created.id)
      setState({
        dataset: { id: created.id, status: 'processing', image_count: 0 },
        job: { id: '', attempt: 1, status: 'processing', progress: 0, error_message: null, max_frames: 0, max_decoded_bytes: 0 },
      })
      setVideo(null)
      setAnnotation(null)
      onMessage?.('Đã nhận video. Backend đang giải mã và kiểm tra annotation.')
      await refresh(created.id)
    } catch (error) {
      onMessage?.((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const retry = async () => {
    if (!datasetId) return
    setBusy(true)
    try {
      await request(`/projects/${session.projectId}/datasets/${datasetId}/import/retry`, { method: 'POST' })
      notifiedReady.current = ''
      onMessage?.('Đã tạo attempt mới cho video import.')
      await refresh(datasetId)
    } catch (error) {
      onMessage?.((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const status = state?.job.status
  return (
    <div className="video-import">
      <div className="video-import-heading">
        <div><strong>Upload video</strong><span className="video-experimental">Thử nghiệm</span></div>
        <small>MP4, MOV hoặc AVI · tối đa 500 MB · annotation tùy chọn</small>
      </div>
      <form onSubmit={upload} className="video-import-form">
        <label>Video<input type="file" accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo" onChange={event => setVideo(event.target.files?.[0] || null)} /></label>
        <label>Annotation<input type="file" accept={annotationAccept} onChange={event => setAnnotation(event.target.files?.[0] || null)} /></label>
        <button disabled={!video || busy}>Upload & xử lý</button>
      </form>
      {state && (
        <div className={`video-import-status video-import-${status}`}>
          <div><strong>Attempt {state.job.attempt}</strong><span>{status} · {state.job.progress}%</span></div>
          <progress max={100} value={state.job.progress} />
          {state.job.error_message && <p>{state.job.error_message}</p>}
          {(status === 'failed' || status === 'interrupted') && <button type="button" className="btn-outline" disabled={busy} onClick={retry}>Thử lại</button>}
        </div>
      )}
    </div>
  )
}
