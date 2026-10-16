import { FormEvent, PointerEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import type { Dataset, FaceResultDetail, FaceResultSummary, FaceRun, Point, Session } from './types'
import { RULE_CODES, VF50_EDGES, VF50_GROUPS } from './vf50'
import TemporalPanel from '../temporal/TemporalPanel'
import TemporalRunOptions from '../temporal/TemporalRunOptions'
import './face.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
type PointDisplayMode = 'suspect' | 'all'

function savedSession(): Session | null {
  try {
    const raw = localStorage.getItem('landmark-session')
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function pointAt(points: Record<string, Point>, id: number): Point | undefined {
  return points[String(id)]
}

function usable(point?: Point): point is Point {
  return !!point && point.state !== 'outside' && point.x !== null && point.y !== null
}

function suspiciousPointIds(result: FaceResultDetail): number[] {
  const ids = new Set<number>()
  result.violations.forEach(item => item.points.forEach(id => ids.add(id)))
  Object.entries(result.quality?.point_evaluations || {}).forEach(([id, evaluation]) => {
    if (evaluation.exceeded || evaluation.severity === 'CRITICAL' || evaluation.severity === 'WARNING') ids.add(Number(id))
  })
  return [...ids].filter(id => id >= 0 && id < 50).sort((a, b) => a - b)
}

function pointGroup(id: number): string {
  const group = VF50_GROUPS.find(([, start, end]) => id >= start && id <= end)
  return group?.[0] || 'VF50'
}

export default function FaceLandmarkPage() {
  const session = savedSession()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [runs, setRuns] = useState<FaceRun[]>([])
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [manifestFile, setManifestFile] = useState<File | null>(null)
  const [temporalEnabled, setTemporalEnabled] = useState(false)
  const [temporalWindow, setTemporalWindow] = useState(15)
  const [activeTemporal, setActiveTemporal] = useState(false)
  const [runId, setRunId] = useState('')
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
  const request = async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await fetch(`${API}${path}`, {
      ...init,
      headers: { ...authHeaders(), ...(init.headers || {}) },
    })
    if (!response.ok) {
      const detail = (await response.json().catch(() => null))?.detail
      throw new Error(detail || `Request failed (${response.status})`)
    }
    return response.json()
  }

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
      if (manifestFile) body.append('sequence_manifest_file', manifestFile)
      const data = await request<Dataset>(`/projects/${session.projectId}/datasets`, { method: 'POST', body })
      setDatasetId(data.id)
      setZipFile(null)
      setAnnotationFile(null)
      setManifestFile(null)
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
      const run = await request<{ id: string }>(
        `/projects/${session.projectId}/runs?dataset_id=${datasetId}&temporal_enabled=${temporalEnabled}&window_size=${temporalWindow}`,
        { method: 'POST' },
      )
      setRunId(run.id)
      setActiveTemporal(temporalEnabled)
      setMessage('Face run đã bắt đầu.')
      const poll = async () => {
        const state = await request<{ status: string; progress: number; error_message?: string }>(
          `/projects/${session.projectId}/runs/${run.id}`,
        )
        setMessage(`Face run: ${state.status} (${state.progress}%)`)
        if (state.status === 'completed') {
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

  const imageUrl = useMemo(() => {
    if (!session || !selected || !datasetId) return ''
    const path = selected.file_name.split('/').map(encodeURIComponent).join('/')
    return `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${path}?token=${encodeURIComponent(session.recoveryKey)}`
  }, [session?.projectId, selected?.id, datasetId])

  const openRun = async (targetRun: FaceRun) => {
    setRunId(targetRun.id)
    setDatasetId(targetRun.dataset_id)
    setActiveTemporal(Boolean(targetRun.temporal_enabled))
    setTemporalEnabled(Boolean(targetRun.temporal_enabled))
    setTemporalWindow(targetRun.temporal_window_size || 15)
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

  if (!session) {
    return (
      <main className="tool-page face-page">
        <header>
          <div>
            <h1>Face Landmark VF50</h1>
            <small>Chưa kết nối dự án kiểm định.</small>
          </div>
        </header>

        <section className="face-empty" style={{ margin: '40px auto' }}>
          <div style={{ width: 56, height: 56, margin: '0 auto 16px', borderRadius: '12px', background: 'var(--blue-subtle)', color: 'var(--blue-primary)', display: 'grid', placeItems: 'center' }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
          </div>
          <h2 style={{ fontSize: '1.3rem', margin: '0 0 8px' }}>Chưa có phiên làm việc Project</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '440px', margin: '0 auto 20px' }}>
            Face Landmark dùng chung Project và Recovery key với Human Pose. Vui lòng mở hoặc tạo project để tiếp tục.
          </p>
          <Link className="face-link-button" to="/pose">Mở hoặc tạo project ở Human Pose →</Link>
        </section>
      </main>
    )
  }

  return (
    <main className="tool-page face-page">
      <header>
        <div>
          <h1>Face Landmark VF50</h1>
          <small>{session.projectName} · Project <code>{session.projectId}</code></small>
        </div>
        <button className="btn-outline" onClick={() => setGuidelineOpen(true)}>Guideline VF50</button>
      </header>

      {message && <p className="notice">{message}</p>}

      <section className="face-setup">
        <div>
          <h2>1. Import dữ liệu Face</h2>
          <form onSubmit={upload} className="face-upload-form">
            <label>ZIP ảnh<input type="file" accept=".zip" onChange={event => setZipFile(event.target.files?.[0] || null)} /></label>
            <label>Nhãn tùy chọn<input type="file" accept=".xml,.json" onChange={event => setAnnotationFile(event.target.files?.[0] || null)} /></label>
            <label>Sequence manifest tùy chọn<input type="file" accept=".json" onChange={event => setManifestFile(event.target.files?.[0] || null)} /></label>
            <button disabled={!zipFile || busy}>Upload Face dataset</button>
          </form>
        </div>
        <div>
          <h2>2. Chạy kiểm định</h2>
          <select value={datasetId} onChange={event => setDatasetId(event.target.value)}>
            <option value="">Chọn Face dataset</option>
            {datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name} ({dataset.image_count} ảnh)</option>)}
          </select>
          {datasetId && <TemporalRunOptions session={session} datasetId={datasetId} enabled={temporalEnabled} windowSize={temporalWindow} onEnabled={setTemporalEnabled} onWindow={setTemporalWindow} />}
          <button disabled={!datasetId || busy} onClick={startRun}>Chạy MediaPipe + 14 luật</button>
          {runs.length > 0 && (
            <div className="face-run-history">
              <span>Run gần đây:</span>
              {runs.slice(0, 5).map(run => (
                <button key={run.id} className="btn-outline" disabled={run.status !== 'completed'} onClick={() => openRun(run)}>
                  {run.dataset_name} · {run.status}{run.temporal_enabled ? ' · Temporal' : ''} · {run.id.slice(0, 8)}
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      {runId && (
        <section>
          <div className="face-toolbar">
            <div>
              <select value={sortBy} onChange={event => setSortBy(event.target.value)}>
                <option value="severity">Lỗi nặng trước</option><option value="frame_index">Theo frame</option><option value="nme">NME cao trước</option>
              </select>
              <select value={filterRule} onChange={event => setFilterRule(event.target.value)}>
                <option value="">Tất cả luật</option>{RULE_CODES.map(code => <option key={code}>{code}</option>)}
              </select>
              <select value={resultKind} onChange={event => setResultKind(event.target.value)}>
                <option value="annotated_face">Mặt có nhãn</option>
                <option value="unlabeled_face">Mặt chưa nhãn ({unlabeledCount})</option>
                <option value="all">Tất cả mặt</option>
              </select>
            </div>
            <div className="face-actions">
              <button className="btn-success" onClick={batchFix}>Preview & xác nhận batch</button>
              <button className="btn-export" onClick={() => download('cleaned-json')}>JSON</button>
              <button className="btn-export" onClick={() => download('cvat-xml')}>CVAT XML</button>
              <button className="btn-outline" onClick={() => download('csv')}>CSV</button>
              <button className="btn-outline" onClick={() => download('checklist')}>Checklist</button>
            </div>
          </div>

          <div className="face-workspace">
            <aside className="face-result-list">
              {results.map(item => (
                <button key={item.id} className={`face-result ${selected?.id === item.id ? 'active' : ''}`} onClick={() => chooseResult(item)}>
                  <span><b>Frame {item.frame_index}</b> · Face {item.face_index + 1}</span>
                  <small>{item.file_name}</small>
                  <span>
                    <strong>{item.error_count} lỗi{item.violation_count !== undefined ? ` / ${item.violation_count} vi phạm` : ''}</strong> ·
                    {item.nme !== null && item.nme !== undefined ? ` NME ${(item.nme * 100).toFixed(2)}%` : ' NME N/A'} ·
                    {item.status}
                  </span>
                  {item.has_outlier_point && <span className="face-tag" style={{ background: '#fef3c7', color: '#92400e', padding: '1px 5px', borderRadius: '3px', fontSize: '11px', display: 'inline-block', marginTop: '2px' }}>⚠️ Outlier</span>}
                  {item.result_kind === 'unlabeled_face' && <span className="face-tag" style={{ background: '#e0e7ff', color: '#3730a3', padding: '1px 5px', borderRadius: '3px', fontSize: '11px', display: 'inline-block', marginTop: '2px' }}>Chưa gán nhãn</span>}
                  {item.case_type !== 'normal' && <em>{item.case_label}</em>}
                </button>
              ))}
              {!results.length && <p className="face-muted">Không có frame nào cần review.</p>}
            </aside>

            <div className="face-inspector">
              {!selected ? <p className="face-muted">Chọn một frame để kiểm tra.</p> : (
                <>
                  <div className="face-view-controls">
                    <span className="face-control-label">Hiển thị:</span>
                    <div className="face-display-mode" role="group" aria-label="Chế độ hiển thị điểm">
                      <button className={pointDisplayMode === 'suspect' ? 'active' : ''} onClick={() => setPointDisplayMode('suspect')}>Điểm nghi ngờ</button>
                      <button className={pointDisplayMode === 'all' ? 'active' : ''} onClick={() => setPointDisplayMode('all')}>Tất cả điểm</button>
                    </div>
                    <label><input type="checkbox" checked={showHuman} onChange={event => setShowHuman(event.target.checked)} /> Human</label>
                    <label><input type="checkbox" checked={showModel} onChange={event => setShowModel(event.target.checked)} /> Model</label>
                    <label><input type="checkbox" checked={showVectors} onChange={event => setShowVectors(event.target.checked)} /> Vector</label>
                    <label><input type="checkbox" checked={showSkeleton} onChange={event => setShowSkeleton(event.target.checked)} /> Skeleton</label>
                    <button className="btn-outline" onClick={() => setScale(value => Math.min(4, value + .25))}>+</button>
                    <button className="btn-outline" onClick={() => setScale(value => Math.max(.5, value - .25))}>−</button>
                    <button className="btn-outline" onClick={() => { setScale(1); setOffset({ x: 0, y: 0 }) }}>Reset</button>
                  </div>
                  <div className="face-metrics-bar" style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', padding: '6px 12px', background: '#f1f5f9', borderRadius: '6px', margin: '6px 0', fontSize: '13px' }}>
                    <span>IOD: <b>{selected.iod?.toFixed(1)} px</b> ({selected.iod_source || 'model'})</span>
                    <span>NME: <b>{selected.nme !== null && selected.nme !== undefined ? `${(selected.nme * 100).toFixed(2)}%` : 'N/A'}</b></span>
                    <span>Điểm cần xem: <b>{suspiciousPointIds(selected).length}</b></span>
                    {selected.quality?.has_outlier_point && <span style={{ color: '#b91c1c', fontWeight: 600 }}>⚠️ Có điểm lệch nặng</span>}
                  </div>
                  <div className="face-canvas-stage">
                    <svg
                      viewBox={`0 0 ${selected.width || 1280} ${selected.height || 720}`}
                      style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }}
                      onPointerDown={onPointerDown}
                      onPointerMove={onPointerMove}
                      onPointerUp={() => { drag.current = null }}
                      onPointerCancel={() => { drag.current = null }}
                    >
                      <image href={imageUrl} width={selected.width || 1280} height={selected.height || 720} />
                      {showSkeleton && VF50_EDGES.map(([from, to]) => {
                        const humanA = pointAt(selected.human_keypoints, from), humanB = pointAt(selected.human_keypoints, to)
                        const modelA = pointAt(selected.model_keypoints, from), modelB = pointAt(selected.model_keypoints, to)
                        return <g key={`${from}-${to}`}>
                          {showHuman && usable(humanA) && usable(humanB) && <line x1={humanA.x!} y1={humanA.y!} x2={humanB.x!} y2={humanB.y!} className="face-edge face-edge-human" style={{ strokeWidth: 0.9 / scale }} />}
                          {showModel && pointDisplayMode === 'all' && usable(modelA) && usable(modelB) && <line x1={modelA.x!} y1={modelA.y!} x2={modelB.x!} y2={modelB.y!} className="face-edge face-edge-model" style={{ strokeWidth: 0.9 / scale }} />}
                        </g>
                      })}
                      {showVectors && Array.from({ length: 50 }, (_, id) => {
                        const human = pointAt(selected.human_keypoints, id), model = pointAt(selected.model_keypoints, id)
                        return renderedPointIds.has(id) && usable(human) && usable(model) ? <line key={`v-${id}`} x1={human.x!} y1={human.y!} x2={model.x!} y2={model.y!} className="face-vector" style={{ strokeWidth: 0.9 / scale }} /> : null
                      })}
                      {showHuman && Object.values(selected.human_keypoints).map(point => {
                        if (!usable(point)) return null
                        const isSuspect = pointDisplayMode === 'suspect' && renderedPointIds.has(point.id)
                        return <circle key={`h-${point.id}`} cx={point.x!} cy={point.y!} r={(isSuspect ? 1.8 : 1.1) / scale} strokeWidth={(isSuspect ? 0.65 : 0) / scale} className={`face-point-human${isSuspect ? ' face-point-human-suspect' : ''}`} onClick={event => { event.stopPropagation(); setSelectedPoint(point.id) }} />
                      })}
                      {showModel && Object.values(selected.model_keypoints).map(point => usable(point) && renderedPointIds.has(point.id) ? <circle key={`m-${point.id}`} cx={point.x!} cy={point.y!} r={1.8 / scale} strokeWidth={0.65 / scale} className="face-point-model" onClick={event => { event.stopPropagation(); setSelectedPoint(point.id) }} /> : null)}
                      {pointDisplayMode === 'suspect' && Array.from(renderedPointIds).map(id => {
                        const point = pointAt(selected.human_keypoints, id) || pointAt(selected.model_keypoints, id)
                        const labelOnRight = id % 2 === 1
                        return usable(point) ? <text key={`id-${id}`} x={point.x! + (labelOnRight ? 3 : -3) / scale} y={point.y! + 1.5 / scale} textAnchor={labelOnRight ? 'start' : 'end'} fontSize={3.5 / scale} strokeWidth={0.35 / scale} className="face-suspect-id">{id}</text> : null
                      })}
                    </svg>
                  </div>

                  <div className="face-detail-grid">
                    <div className="face-violations">
                      <h3>Vi phạm ({selected.violations.length})</h3>
                      {selected.violations.map(item => (
                        <button key={`${item.code}-${item.message}`} onClick={() => item.points[0] !== undefined && setSelectedPoint(item.points[0])}>
                          <b>{item.code} · {item.rule_name}</b><span>{item.message}</span><small>{item.gl_section}</small>
                        </button>
                      ))}
                      {!selected.violations.length && <p className="face-ok">Không phát hiện vi phạm rule.</p>}
                    </div>
                    <div className="face-point-review">
                      <h3>Duyệt điểm {selectedPoint} · {pointGroup(selectedPoint)}</h3>
                      <p>Nguồn hiện tại: <b>{selected.point_sources[String(selectedPoint)] || (Object.keys(selected.human_keypoints).length ? 'human' : 'model')}</b></p>
                      {selectedEvaluation && <p className="face-point-metric">Lệch <b>{selectedEvaluation.dist_px.toFixed(1)} px</b> · <b>{selectedEvaluation.norm_error_pct.toFixed(1)}% IOD</b>{selectedEvaluation.is_anchor ? ' · Điểm neo' : ''}</p>}
                      <button className="btn-success" disabled={!pointAt(selected.human_keypoints, selectedPoint)} onClick={() => setSource(selectedPoint, 'human')}>Giữ Human</button>
                      <button disabled={!usable(pointAt(selected.model_keypoints, selectedPoint))} onClick={() => setSource(selectedPoint, 'model')}>Lấy Model</button>
                      <div className="face-review-list">
                        {reviewPointIds.map(id => {
                          const source = selected.point_sources[String(id)] || (Object.keys(selected.human_keypoints).length ? 'human' : 'model')
                          const evaluation = selected.quality?.point_evaluations?.[String(id)]
                          return <button key={id} className={`${id === selectedPoint ? 'active' : ''} source-${source}`} onClick={() => setSelectedPoint(id)}>
                            <b>#{id} · {pointGroup(id)}</b>
                            <span>{evaluation ? `${evaluation.dist_px.toFixed(1)} px · ${evaluation.norm_error_pct.toFixed(1)}% IOD` : 'Vi phạm guideline'}</span>
                          </button>
                        })}
                        {!reviewPointIds.length && <p className="face-ok">Không có điểm vượt ngưỡng hoặc nằm trong vi phạm.</p>}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      )}

      {runId && activeTemporal && datasetId && <TemporalPanel session={session} runId={runId} datasetId={datasetId} />}

      {guidelineOpen && (
        <div className="face-modal-backdrop" onClick={() => setGuidelineOpen(false)}>
          <div className="face-modal" onClick={event => event.stopPropagation()}>
            <button className="face-modal-close" onClick={() => setGuidelineOpen(false)}>×</button>
            <h2>VF50: 7 nhóm, 50 điểm</h2>
            {VF50_GROUPS.map(([name, start, end]) => <p key={name}><b>{name}</b>: điểm {start}–{end}</p>)}
            <p>Điểm neo dùng dung sai 3% IOD; điểm contour dùng 5% IOD; NME mục tiêu ≤ 3.5%.</p>
            <p>Luôn kiểm tra trạng thái Visible, Occluded và Outside trước khi chọn đề xuất AI.</p>
          </div>
        </div>
      )}
    </main>
  )
}
