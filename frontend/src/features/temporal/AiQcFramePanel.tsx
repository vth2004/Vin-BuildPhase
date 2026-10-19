import { useEffect, useMemo, useState } from 'react'
import { VF50_EDGES } from '../face-landmark-vf50/vf50'
import './temporal.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
const POSE_EDGES: [number, number][] = [[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,7],[6,12],[7,13],[12,13],[6,8],[8,10],[7,9],[9,11],[12,14],[14,16],[13,15],[15,17]]
type Session = { projectId: string; recoveryKey: string }
type Point = { id: number; x?: number | null; y?: number | null; state?: string; confidence?: number | null }
type Draft = { id: string; image_id: string; file_name: string; frame_index: number | null; instance_index: number; track_id?: string | null; keypoints: Point[]; qc: { flags?: string[] } }
type TPoint = { image_id: string; internal_track_id: string; keypoint_id: number; temporal_x?: number | null; temporal_y?: number | null; estimate_status: string; unavailable_reason?: string | null; temporal_quality: number; support_count: number; residual?: number | null }
type Frame = { imageId: string; fileName: string; frameIndex: number | null; drafts: Draft[] }

export default function AiQcFramePanel({ session, runId, datasetId, feature }: { session: Session; runId: string; datasetId: string; feature: 'pose' | 'face' }) {
  const [drafts, setDrafts] = useState<Draft[]>([]), [temporal, setTemporal] = useState<TPoint[]>([])
  const [cursor, setCursor] = useState(0), [selected, setSelected] = useState<{ draft: Draft; point: Point } | null>(null)
  const [layer, setLayer] = useState<'temporal' | 'model'>('temporal'), [size, setSize] = useState({ width: 1280, height: 720 })
  const [message, setMessage] = useState('')
  const headers = { Authorization: `Bearer ${session.recoveryKey}` }

  useEffect(() => {
    let active = true
    const load = async () => {
      const response = await fetch(`${API}/projects/${session.projectId}/runs/${runId}/ai-drafts`, { headers })
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `HTTP ${response.status}`)
      const payload = await response.json() as { items: Draft[] }, points: TPoint[] = []
      for (let page = 1; ; page += 1) {
        const resultResponse = await fetch(`${API}/projects/${session.projectId}/runs/${runId}/temporal/results?page=${page}&page_size=200`, { headers })
        if (!resultResponse.ok) throw new Error((await resultResponse.json().catch(() => null))?.detail || `HTTP ${resultResponse.status}`)
        const result = await resultResponse.json() as { total: number; items: TPoint[] }; points.push(...result.items)
        if (points.length >= result.total) break
      }
      if (active) { setDrafts(payload.items); setTemporal(points); setCursor(0); setSelected(null) }
    }
    load().catch(error => setMessage(error.message)); return () => { active = false }
  }, [runId])

  const frames = useMemo(() => {
    const grouped = new Map<string, Frame>()
    drafts.forEach(draft => { const value = grouped.get(draft.image_id) || { imageId: draft.image_id, fileName: draft.file_name, frameIndex: draft.frame_index, drafts: [] }; value.drafts.push(draft); grouped.set(draft.image_id, value) })
    return [...grouped.values()].sort((a, b) => (a.frameIndex ?? Number.MAX_SAFE_INTEGER) - (b.frameIndex ?? Number.MAX_SAFE_INTEGER) || a.fileName.localeCompare(b.fileName))
  }, [drafts])
  const frame = frames[cursor], tMap = useMemo(() => new Map(temporal.map(point => [`${point.image_id}:${point.internal_track_id}:${point.keypoint_id}`, point])), [temporal])
  const tPoint = (draft: Draft, point: Point) => tMap.get(`${draft.image_id}:${draft.track_id || ''}:${point.id}`)
  const shown = (draft: Draft, point?: Point) => {
    if (!point) return null; const estimate = tPoint(draft, point)
    if (layer === 'temporal' && estimate?.estimate_status === 'available' && estimate.temporal_x != null && estimate.temporal_y != null) return { x: estimate.temporal_x, y: estimate.temporal_y, temporal: true }
    return point.x != null && point.y != null ? { x: point.x, y: point.y, temporal: false } : null
  }
  const imageUrl = frame ? `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${frame.fileName.split('/').map(encodeURIComponent).join('/')}?token=${encodeURIComponent(session.recoveryKey)}` : ''
  const downloadCvat = async () => { try { const response = await fetch(`${API}/projects/${session.projectId}/runs/${runId}/ai-drafts/export/cvat`, { headers }); if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `HTTP ${response.status}`); const url = URL.createObjectURL(await response.blob()), a = document.createElement('a'); a.href = url; a.download = `ai-draft-cvat-${runId.slice(0, 8)}.xml`; a.click(); URL.revokeObjectURL(url) } catch (error) { setMessage((error as Error).message) } }
  const selectedTemporal = selected ? tPoint(selected.draft, selected.point) : undefined, edges = feature === 'face' ? VF50_EDGES : POSE_EDGES
  const scale = Math.max(1, Math.min(size.width, size.height) / 720)

  return <section className="temporal-panel ai-qc-panel">
    <header className="temporal-header"><div><h2>AI draft <span>Không có annotation nguồn</span></h2><p>Xem toàn bộ landmark theo từng frame. Nhấn một điểm để xem chất lượng temporal.</p></div><button onClick={downloadCvat}>Tải annotation cho CVAT</button></header>
    {message && <p className="notice">{message}</p>}
    <div className="ai-qc-toolbar"><div className="ai-qc-layer"><button className={layer === 'temporal' ? 'active' : ''} onClick={() => setLayer('temporal')}>Temporal (ưu tiên)</button><button className={layer === 'model' ? 'active' : ''} onClick={() => setLayer('model')}>Model gốc</button></div><span>Tím: Temporal · Đỏ: model/fallback</span></div>
    {!frame ? <p>Run chưa có AI draft để hiển thị.</p> : <div className="ai-qc-workspace">
      <aside className="ai-qc-frames">{frames.map((item, index) => <button key={item.imageId} className={index === cursor ? 'active' : ''} onClick={() => { setCursor(index); setSelected(null) }}><b>Frame {item.frameIndex ?? index}</b><small>{item.fileName}</small><span>{item.drafts.length} instance</span></button>)}</aside>
      <div className="temporal-inspector"><div className="temporal-nav"><button disabled={!cursor} onClick={() => { setCursor(cursor - 1); setSelected(null) }}>← Trước</button><b>Frame {frame.frameIndex ?? cursor} · {frame.drafts.length} instance</b><button disabled={cursor >= frames.length - 1} onClick={() => { setCursor(cursor + 1); setSelected(null) }}>Sau →</button></div>
        <div className="temporal-canvas ai-qc-canvas"><img src={imageUrl} onLoad={event => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} /><svg viewBox={`0 0 ${size.width} ${size.height}`}>
          {frame.drafts.flatMap(draft => edges.map(([from, to], index) => { const a = shown(draft, draft.keypoints.find(point => point.id === from)), b = shown(draft, draft.keypoints.find(point => point.id === to)); return a && b ? <line key={`${draft.id}:e:${index}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={a.temporal && b.temporal ? 'ai-edge temporal' : 'ai-edge model'} /> : null }))}
          {frame.drafts.flatMap(draft => draft.keypoints.map(point => { const value = shown(draft, point); return value ? <g key={`${draft.id}:p:${point.id}`} className="ai-point" onClick={() => setSelected({ draft, point })}><circle cx={value.x} cy={value.y} r={6 * scale} className={value.temporal ? 'temporal-estimate' : 'temporal-model'} /><text x={value.x + 9 * scale} y={value.y - 8 * scale} className={value.temporal ? 'temporal-point-label estimate' : 'temporal-point-label model'} style={{ fontSize: 14 * scale, strokeWidth: 4 * scale }}>{point.id}</text></g> : null }))}
        </svg></div>
        {selected && <div className="ai-qc-point-detail"><b>Instance {selected.draft.instance_index + 1} · Point {selected.point.id}</b><span>Model confidence: {selected.point.confidence == null ? 'không có' : `${Math.round(selected.point.confidence * 100)}%`}</span><span>Temporal: {selectedTemporal?.estimate_status === 'available' ? `Q ${Math.round(selectedTemporal.temporal_quality * 100)}% · support ${selectedTemporal.support_count} · residual ${selectedTemporal.residual?.toFixed(4) ?? '—'}` : `không khả dụng · ${selectedTemporal?.unavailable_reason || 'không có kết quả'}`}</span>{Boolean(selected.draft.qc.flags?.length) && <span>Cảnh báo: {selected.draft.qc.flags!.join(', ')}</span>}</div>}
      </div></div>}
    <p className="ai-qc-export-note">File xuất là CVAT for video XML, dùng original frame index và giữ track của từng người/khuôn mặt. Trong CVAT chọn Upload annotations → CVAT for video 1.1.</p>
  </section>
}
