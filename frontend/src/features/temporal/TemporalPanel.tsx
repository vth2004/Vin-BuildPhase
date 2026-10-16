import { useEffect, useMemo, useState } from 'react'

import './temporal.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

type Session = { projectId: string; recoveryKey: string }
type TemporalItem = {
  id: string; file_name: string; frame_index: number; keypoint_id: number
  sequence_id?: string; segment_id: number; internal_track_id: string
  human_x?: number | null; human_y?: number | null; raw_x?: number | null; raw_y?: number | null
  temporal_x?: number | null; temporal_y?: number | null; estimate_status: string; method: string
  unavailable_reason?: string | null; temporal_quality: number; temporal_score?: number | null
  support_count: number; residual?: number | null; normalization_source?: string; time_source: string
  evidence?: string | null; observation_source?: string; supporting_model_count: number
}
type Detail = TemporalItem & { trajectory: Array<{ frame_index: number; timestamp_us?: number | null; file_name?: string | null; point?: { x: number; y: number } }> }
type Summary = { total: number; available: number; coverage: number; warnings: number; unavailable_reasons: Array<{ unavailable_reason: string; count: number }> }

const evidenceLabel: Record<string, string> = {
  model_trajectory: 'Quỹ đạo model', annotation_jitter: 'Nhãn bị giật', face_r13: 'Face R13 (đã gộp)',
}

export default function TemporalPanel({ session, runId, datasetId }: { session: Session; runId: string; datasetId: string }) {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [items, setItems] = useState<TemporalItem[]>([])
  const [selected, setSelected] = useState<Detail | null>(null)
  const [status, setStatus] = useState('all')
  const [evidence, setEvidence] = useState('all')
  const [message, setMessage] = useState('')
  const [size, setSize] = useState({ width: 1280, height: 720 })

  const request = async <T,>(path: string): Promise<T> => {
    const response = await fetch(`${API}${path}`, { headers: { Authorization: `Bearer ${session.recoveryKey}` } })
    if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `HTTP ${response.status}`)
    return response.json()
  }
  const loadDetail = async (id: string) => {
    const detail = await request<Detail>(`/projects/${session.projectId}/runs/${runId}/temporal/results/${id}`)
    setSelected(detail)
  }
  const load = async () => {
    const query = new URLSearchParams({ page_size: '200' })
    if (status !== 'all') query.set('estimate_status', status)
    if (evidence !== 'all') query.set('evidence', evidence)
    const [nextSummary, page] = await Promise.all([
      request<Summary>(`/projects/${session.projectId}/runs/${runId}/temporal/summary`),
      request<{ items: TemporalItem[] }>(`/projects/${session.projectId}/runs/${runId}/temporal/results?${query}`),
    ])
    setSummary(nextSummary)
    setItems(page.items)
    if (page.items.length) await loadDetail(page.items[0].id)
    else setSelected(null)
  }
  useEffect(() => { load().catch(error => setMessage(error.message)) }, [runId, status, evidence])

  const imageUrl = (fileName?: string | null) => fileName
    ? `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${fileName.split('/').map(encodeURIComponent).join('/')}?token=${encodeURIComponent(session.recoveryKey)}`
    : ''
  const selectedIndex = items.findIndex(item => item.id === selected?.id)
  const trajectoryPoints = useMemo(() => selected?.trajectory.filter(item => item.point).map(item => `${item.point!.x},${item.point!.y}`).join(' ') || '', [selected])

  return <section className="temporal-panel">
    <header className="temporal-header">
      <div><h2>Temporal QA <span>Thử nghiệm · chỉ đọc</span></h2><p>So sánh nhãn với chuyển động cùng track qua nhiều frame.</p></div>
      {summary && <div className="temporal-summary">
        <b>{Math.round(summary.coverage * 100)}% coverage</b><span>{summary.available}/{summary.total} điểm</span><span>{summary.warnings} cảnh báo</span>
      </div>}
    </header>
    {message && <p className="notice">{message}</p>}
    <div className="temporal-filters">
      <label>Trạng thái <select value={status} onChange={event => setStatus(event.target.value)}><option value="all">Tất cả</option><option value="available">Có estimate</option><option value="unavailable">Không khả dụng</option></select></label>
      <label>Bằng chứng <select value={evidence} onChange={event => setEvidence(event.target.value)}><option value="all">Tất cả</option><option value="model_trajectory">Quỹ đạo model</option><option value="annotation_jitter">Nhãn bị giật</option><option value="face_r13">Face R13</option></select></label>
      {summary?.unavailable_reasons.map(item => <span className="temporal-reason" key={item.unavailable_reason}>{item.unavailable_reason}: {item.count}</span>)}
    </div>
    <div className="temporal-workspace">
      <aside className="temporal-list">
        {items.map(item => <button key={item.id} className={selected?.id === item.id ? 'active' : ''} onClick={() => loadDetail(item.id).catch(error => setMessage(error.message))}>
          <b>Frame {item.frame_index} · KP {item.keypoint_id}</b><small>{item.file_name}</small>
          <span>{item.evidence ? evidenceLabel[item.evidence] || item.evidence : item.estimate_status === 'available' ? item.method : item.unavailable_reason}</span>
          <em>Q {Math.round(item.temporal_quality * 100)}%{item.temporal_score != null ? ` · Score ${Math.round(item.temporal_score * 100)}%` : ''}</em>
        </button>)}
        {!items.length && <p>Không có điểm phù hợp bộ lọc.</p>}
      </aside>
      <div className="temporal-inspector">
        {!selected ? <p>Chọn một kết quả temporal để xem chi tiết.</p> : <>
          <div className="temporal-nav"><button disabled={selectedIndex <= 0} onClick={() => loadDetail(items[selectedIndex - 1].id)}>← Trước</button><b>Frame {selected.frame_index} · Track {selected.internal_track_id.slice(0, 8)}</b><button disabled={selectedIndex < 0 || selectedIndex >= items.length - 1} onClick={() => loadDetail(items[selectedIndex + 1].id)}>Sau →</button></div>
          <div className="temporal-canvas">
            <img src={imageUrl(selected.file_name)} onLoad={event => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
            <svg viewBox={`0 0 ${size.width} ${size.height}`}>
              {trajectoryPoints && <polyline points={trajectoryPoints} className="temporal-path" />}
              {selected.human_x != null && selected.human_y != null && <circle cx={selected.human_x} cy={selected.human_y} r="7" className="temporal-human" />}
              {selected.raw_x != null && selected.raw_y != null && <circle cx={selected.raw_x} cy={selected.raw_y} r="7" className="temporal-model" />}
              {selected.temporal_x != null && selected.temporal_y != null && <circle cx={selected.temporal_x} cy={selected.temporal_y} r="8" className="temporal-estimate" />}
            </svg>
          </div>
          <div className="temporal-legend"><span className="human">Human</span><span className="model">Model</span><span className="estimate">Temporal</span></div>
          <div className="temporal-metrics">
            <span>Sequence <b>{selected.sequence_id?.slice(0, 8) || '—'}</b></span><span>Segment <b>{selected.segment_id}</b></span>
            <span>Quality <b>{Math.round(selected.temporal_quality * 100)}%</b></span><span>Support <b>{selected.support_count}</b></span>
            <span>Residual <b>{selected.residual?.toFixed(4) ?? '—'}</b></span><span>Normalize <b>{selected.normalization_source || '—'}</b></span>
            <span>Time <b>{selected.time_source}</b></span><span>Models <b>{selected.supporting_model_count}</b></span>
          </div>
          <div className="temporal-timeline">
            {selected.trajectory.map(frame => <button key={frame.frame_index} className={frame.frame_index === selected.frame_index ? 'active' : ''} title={`Frame ${frame.frame_index}`} onClick={() => { const target = items.find(item => item.frame_index === frame.frame_index && item.internal_track_id === selected.internal_track_id && item.keypoint_id === selected.keypoint_id); if (target) loadDetail(target.id).catch(error => setMessage(error.message)) }}>
              {frame.file_name ? <img src={imageUrl(frame.file_name)} loading="lazy" /> : <span>Không có ảnh</span>}<small>{frame.frame_index}</small>
            </button>)}
          </div>
        </>}
      </div>
    </div>
  </section>
}
