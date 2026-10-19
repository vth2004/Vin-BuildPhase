import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, PointerEvent } from 'react'
import { API, requestJson } from '../../shared/api'
import { readSession } from '../../shared/session'
import type { Dataset, FaceResultDetail, FaceResultSummary, FaceRun } from './types'
import { suspiciousPointIds } from './points'
import type { PointDisplayMode } from './points'

export function useFaceWorkspace() {
  const session = readSession()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [runs, setRuns] = useState<FaceRun[]>([])
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [runId, setRunId] = useState('')
  const [aiDraftAvailable, setAiDraftAvailable] = useState(false)
  const [results, setResults] = useState<FaceResultSummary[]>([])
  const [selected, setSelected] = useState<FaceResultDetail | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<number>(0)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [sortBy, setSortBy] = useState('severity')
  const [filterRule, setFilterRule] = useState('')
  const [resultKind, setResultKind] = useState('annotated_face')
  const [unlabeledCount, setUnlabeledCount] = useState(0)
  const [showHuman, setShowHuman] = useState(true)
  const [showModel, setShowModel] = useState(true)
  const [showVectors, setShowVectors] = useState(true)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [pointDisplayMode, setPointDisplayMode] = useState<PointDisplayMode>('suspect')
  const [guidelineOpen, setGuidelineOpen] = useState(false)
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)

  const authHeaders = () => ({ Authorization: `Bearer ${session?.recoveryKey || ''}` })
  const request = <T,>(path: string, init: RequestInit = {}): Promise<T> =>
    requestJson<T>(path, init, session?.recoveryKey)

  const loadProject = async () => {
    if (!session) return
    const data = await request<{ datasets: Dataset[] }>(`/projects/${session.projectId}`)
    const faceDatasets = data.datasets.filter(item => item.schema_id === 'vf_face_landmark50_v1')
    setDatasets(faceDatasets)
    setDatasetId(current => current || faceDatasets[0]?.id || '')
    const faceRuns = await request<FaceRun[]>(`/projects/${session.projectId}/runs?schema_id=vf_face_landmark50_v1`)
    setRuns(faceRuns)
  }

  const loadResults = async (targetRun = runId) => {
    if (!session || !targetRun) return
    const query = new URLSearchParams({ page_size: '200', sort_by: sortBy })
    if (filterRule) query.set('filter_rule', filterRule)
    if (resultKind) query.set('result_kind', resultKind)
    const data = await request<{ items: FaceResultSummary[]; unlabeled_count?: number }>(
      `/projects/${session.projectId}/runs/${targetRun}/face/results?${query}`,
    )
    const actionable = data.items.filter(item =>
      item.error_count > 0
      || (item.violation_count || 0) > 0
      || (item.exceed_count || 0) > 0
      || (item.nme !== null && item.nme > 0.035),
    )
    setResults(actionable)
    if (data.unlabeled_count !== undefined) setUnlabeledCount(data.unlabeled_count)
    if (actionable.length) await chooseResult(actionable[0], targetRun)
    else setSelected(null)
  }

  const chooseResult = async (item: FaceResultSummary, targetRun = runId) => {
    if (!session || !targetRun) return
    const detail = await request<FaceResultDetail>(
      `/projects/${session.projectId}/runs/${targetRun}/face/results/${item.id}`,
    )
    setSelected(detail)
    const firstViolationPoint = detail.violations.flatMap(item => item.points).find(id => id >= 0 && id < 50)
    setSelectedPoint(firstViolationPoint ?? suspiciousPointIds(detail)[0] ?? 0)
    setScale(1)
    setOffset({ x: 0, y: 0 })
  }

  useEffect(() => {
    loadProject().catch(error => setMessage(error.message))
  }, [session?.projectId])

  useEffect(() => {
    if (runId) loadResults().catch(error => setMessage(error.message))
  }, [sortBy, filterRule, resultKind])

  const upload = async (event: FormEvent) => {
    event.preventDefault()
    if (!session || !zipFile) return
    setBusy(true)
    try {
      const body = new FormData()
      body.append('file', zipFile)
      body.append('schema_id', 'vf_face_landmark50_v1')
      if (annotationFile) body.append('annotation_file', annotationFile)
      const data = await request<Dataset>(`/projects/${session.projectId}/datasets`, { method: 'POST', body })
      setDatasetId(data.id)
      setZipFile(null)
      setAnnotationFile(null)
      setMessage(`Đã nhập ${data.image_count} ảnh vào dataset ${data.name}.`)
      await loadProject()
    } catch (error) {
      setMessage((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const startRun = async () => {
    if (!session || !datasetId) return
    setBusy(true)
    setResults([])
    setSelected(null)
    try {
      const run = await request<{ id: string; input_mode: string }>(
        `/projects/${session.projectId}/runs?dataset_id=${datasetId}`,
        { method: 'POST' },
      )
      setRunId(run.id)
      setAiDraftAvailable(false)
      setMessage('Face run đã bắt đầu.')
      const poll = async () => {
        const state = await request<{ status: string; progress: number; error_message?: string; input_mode?: string }>(
          `/projects/${session.projectId}/runs/${run.id}`,
        )
        setMessage(`Face run: ${state.status} (${state.progress}%)`)
        if (state.status === 'completed') {
          setAiDraftAvailable(state.input_mode === 'ai_qc')
          setBusy(false)
          await loadProject()
          await loadResults(run.id)
          return
        }
        if (state.status === 'failed') {
          setBusy(false)
          setMessage(state.error_message || 'Face run thất bại.')
          return
        }
        window.setTimeout(() => poll().catch(error => {
          setBusy(false)
          setMessage(error.message)
        }), 1000)
      }
      await poll()
    } catch (error) {
      setBusy(false)
      setMessage((error as Error).message)
    }
  }

  const setSource = async (pointId: number, source: 'human' | 'model') => {
    if (!session || !selected || !runId) return
    try {
      const detail = await request<FaceResultDetail>(
        `/projects/${session.projectId}/runs/${runId}/face/results/${selected.id}/points/${pointId}`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source }) },
      )
      setSelected(detail)
      setResults(items => items.map(item => item.id === detail.id ? { ...item, status: detail.status } : item))
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  const batchFix = async () => {
    if (!session || !runId) return
    try {
      const base = { only_violated: true, min_nme: 0.035 }
      const preview = await request<{ result_count: number; point_count: number }>(
        `/projects/${session.projectId}/runs/${runId}/face/batch-preview`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(base) },
      )
      if (!preview.point_count) {
        setMessage('Không có điểm nào đủ điều kiện để batch fix.')
        return
      }
      const confirmed = window.confirm(
        `AI đề xuất thay ${preview.point_count} điểm trong ${preview.result_count} kết quả. Bạn xác nhận áp dụng?`,
      )
      if (!confirmed) return
      await request(`/projects/${session.projectId}/runs/${runId}/face/batch-apply`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...base, confirm: true }),
      })
      setMessage(`Đã lưu lựa chọn Model cho ${preview.point_count} điểm. Nhãn gốc không bị thay đổi.`)
      await loadResults()
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  const download = async (format: 'cleaned-json' | 'cvat-xml' | 'csv' | 'checklist') => {
    if (!session || !runId) return
    try {
      const response = await fetch(
        `${API}/projects/${session.projectId}/runs/${runId}/face/export/${format}`,
        { headers: authHeaders() },
      )
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || 'Export failed')
      const blob = await response.blob()
      const disposition = response.headers.get('Content-Disposition') || ''
      const name = disposition.match(/filename="([^"]+)"/)?.[1] || `vf50-${format}`
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = name
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  const downloadAiDraft = async () => {
    if (!session || !runId) return
    try {
      const response = await fetch(`${API}/projects/${session.projectId}/runs/${runId}/ai-drafts/export`, { headers: authHeaders() })
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || 'Export AI draft failed')
      const url = URL.createObjectURL(await response.blob())
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `face_ai_draft_${runId}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  const imageUrl = useMemo(() => {
    if (!session || !selected || !datasetId) return ''
    const path = selected.file_name.split('/').map(encodeURIComponent).join('/')
    return `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${path}?token=${encodeURIComponent(session.recoveryKey)}`
  }, [session?.projectId, selected?.id, datasetId])

  const openRun = async (targetRun: FaceRun) => {
    setRunId(targetRun.id)
    setAiDraftAvailable(targetRun.input_mode === 'ai_qc')
    setDatasetId(targetRun.dataset_id)
    setMessage(`Đã mở run ${targetRun.id.slice(0, 8)}: ${targetRun.status}.`)
    await loadResults(targetRun.id)
  }

  const reviewedPoints = selected?.reviewed_keypoints || {}
  const reviewPointIds = useMemo(() => {
    if (!selected) return []
    return pointDisplayMode === 'all' ? Array.from({ length: 50 }, (_, id) => id) : suspiciousPointIds(selected)
  }, [selected, pointDisplayMode])
  const renderedPointIds = useMemo(() => {
    const ids = new Set(reviewPointIds)
    if (pointDisplayMode === 'all' && selectedPoint >= 0 && selectedPoint < 50) ids.add(selectedPoint)
    return ids
  }, [reviewPointIds, selectedPoint, pointDisplayMode])
  const selectedEvaluation = selected?.quality?.point_evaluations?.[String(selectedPoint)]
  const onPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = { x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y }
  }
  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return
    setOffset({ x: drag.current.ox + event.clientX - drag.current.x, y: drag.current.oy + event.clientY - drag.current.y })
  }

  return { session, datasets, datasetId, setDatasetId, runs, zipFile, setZipFile, setAnnotationFile, runId, aiDraftAvailable, results, selected, selectedPoint, setSelectedPoint, message, busy, sortBy, setSortBy, filterRule, setFilterRule, resultKind, setResultKind, unlabeledCount, showHuman, setShowHuman, showModel, setShowModel, showVectors, setShowVectors, showSkeleton, setShowSkeleton, pointDisplayMode, setPointDisplayMode, guidelineOpen, setGuidelineOpen, scale, setScale, offset, setOffset, drag, chooseResult, upload, startRun, setSource, batchFix, download, downloadAiDraft, imageUrl, openRun, reviewPointIds, renderedPointIds, selectedEvaluation, onPointerDown, onPointerMove }
}
