import { useEffect, useMemo, useState } from 'react'

import './temporal.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

type Session = { projectId: string; recoveryKey: string }
type TemporalItem = {
  id: string; file_name: string; frame_index: number; keypoint_id: number
  annotation_id?: string | null
  sequence_id?: string; segment_id: number; internal_track_id: string
  human_x?: number | null; human_y?: number | null; raw_x?: number | null; raw_y?: number | null
  temporal_x?: number | null; temporal_y?: number | null; estimate_status: string; method: string
  unavailable_reason?: string | null; temporal_quality: number; temporal_score?: number | null
  support_count: number; residual?: number | null; normalization_source?: string; time_source: string
  evidence?: string | null; observation_source?: string; supporting_model_count: number
}
type Detail = TemporalItem & { trajectory: Array<{ frame_index: number; timestamp_us?: number | null; file_name?: string | null; point?: { x: number; y: number } }> }
type ReviewSource = 'human' | 'temporal'
type Review = { id: string; temporal_result_id: string; annotation_id: string; keypoint_id: number; source: ReviewSource; selected_x: number; selected_y: number; reviewed_at: number }
type Summary = { total: number; available: number; coverage: number; warnings: number; reviewed_total: number; reviewed_human: number; reviewed_temporal: number; unavailable_reasons: Array<{ unavailable_reason: string; count: number }> }
type ResultPage = { total: number; page: number; page_size: number; items: TemporalItem[] }

const PAGE_SIZE = 200

const evidenceLabel: Record<string, string> = {
  model_trajectory: 'Quỹ đạo model', annotation_jitter: 'Nhãn bị giật', face_r13: 'Face R13 (đã gộp)',
}

export default function TemporalPanel({ session, runId, datasetId }: { session: Session; runId: string; datasetId: string }) {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [items, setItems] = useState<TemporalItem[]>([])
  const [selected, setSelected] = useState<Detail | null>(null)
  const [status, setStatus] = useState('all')
  const [evidence, setEvidence] = useState('all')
  const [reviewFilter, setReviewFilter] = useState('all')
  const [reviews, setReviews] = useState<Review[]>([])
  const [reviewBusy, setReviewBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [size, setSize] = useState({ width: 1280, height: 720 })
  const [pageNumber, setPageNumber] = useState(1)
  const [resultTotal, setResultTotal] = useState(0)

  const request = async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await fetch(`${API}${path}`, { ...init, headers: { Authorization: `Bearer ${session.recoveryKey}`, ...(init.headers || {}) } })
    if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `HTTP ${response.status}`)
    if (response.status === 204) return undefined as T
    return response.json()
  }
  const loadDetail = async (id: string) => {
    const detail = await request<Detail>(`/projects/${session.projectId}/runs/${runId}/temporal/results/${id}`)
    setSelected(detail)
  }
  const load = async (targetPage = pageNumber, selectEdge: 'first' | 'last' = 'first') => {
    const query = new URLSearchParams({ page_size: String(PAGE_SIZE), page: String(targetPage) })
    if (status !== 'all') query.set('estimate_status', status)
    if (evidence !== 'all') query.set('evidence', evidence)
    const [nextSummary, page, reviewPage] = await Promise.all([
      request<Summary>(`/projects/${session.projectId}/runs/${runId}/temporal/summary`),
      request<ResultPage>(`/projects/${session.projectId}/runs/${runId}/temporal/results?${query}`),
      request<{ items: Review[] }>(`/projects/${session.projectId}/runs/${runId}/temporal/reviews`),
    ])
    setSummary(nextSummary)
    setItems(page.items)
    setPageNumber(page.page)
    setResultTotal(page.total)
    setReviews(reviewPage.items)
    if (page.items.length) await loadDetail(page.items[selectEdge === 'last' ? page.items.length - 1 : 0].id)
    else setSelected(null)
  }
  useEffect(() => { setPageNumber(1); load(1).catch(error => setMessage(error.message)) }, [runId, status, evidence])

  const imageUrl = (fileName?: string | null) => fileName
    ? `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${fileName.split('/').map(encodeURIComponent).join('/')}?token=${encodeURIComponent(session.recoveryKey)}`
    : ''
  const reviewKey = (item: Pick<TemporalItem, 'annotation_id' | 'keypoint_id'>) => `${item.annotation_id || ''}:${item.keypoint_id}`
  const reviewsByKey = useMemo(() => new Map(reviews.map(review => [`${review.annotation_id}:${review.keypoint_id}`, review])), [reviews])
  const visibleItems = useMemo(() => items.filter(item => {
    const review = reviewsByKey.get(reviewKey(item))
    return reviewFilter === 'all' || (reviewFilter === 'unreviewed' ? !review : review?.source === reviewFilter)
  }), [items, reviewsByKey, reviewFilter])
  const selectedIndex = visibleItems.findIndex(item => item.id === selected?.id)
  const selectedReview = selected ? reviewsByKey.get(reviewKey(selected)) : undefined
  const trajectoryPoints = useMemo(() => selected?.trajectory.filter(item => item.point).map(item => `${item.point!.x},${item.point!.y}`).join(' ') || '', [selected])
  const overlayScale = Math.max(1, Math.min(size.width, size.height) / 720)
  const totalPages = Math.max(1, Math.ceil(resultTotal / PAGE_SIZE))

  const goPrevious = async () => {
    if (selectedIndex > 0) return loadDetail(visibleItems[selectedIndex - 1].id)
    if (pageNumber > 1) await load(pageNumber - 1, 'last')
  }
  const goNext = async () => {
    if (selectedIndex >= 0 && selectedIndex < visibleItems.length - 1) return loadDetail(visibleItems[selectedIndex + 1].id)
    if (pageNumber < totalPages) await load(pageNumber + 1, 'first')
  }

  const saveReview = async (source: ReviewSource) => {
    if (!selected) return
    setReviewBusy(true); setMessage('')
    try {
      await request<Review>(`/projects/${session.projectId}/runs/${runId}/temporal/results/${selected.id}/review`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source }),
      })
      await load()
      await loadDetail(selected.id)
      setMessage(source === 'temporal' ? 'Đã chọn điểm Temporal.' : 'Đã giữ điểm Human.')
    } catch (error) { setMessage((error as Error).message) } finally { setReviewBusy(false) }
  }
  const clearReview = async () => {
    if (!selected) return
    setReviewBusy(true); setMessage('')
    try {
      await request<void>(`/projects/${session.projectId}/runs/${runId}/temporal/results/${selected.id}/review`, { method: 'DELETE' })
      await load()
      await loadDetail(selected.id)
      setMessage('Đã bỏ lựa chọn review.')
    } catch (error) { setMessage((error as Error).message) } finally { setReviewBusy(false) }
  }
  const downloadAudit = async (format: 'json' | 'csv') => {
    try {
      const response = await fetch(`${API}/projects/${session.projectId}/runs/${runId}/temporal/export/audit.${format}`, {
        headers: { Authorization: `Bearer ${session.recoveryKey}` },
      })
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `HTTP ${response.status}`)
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `temporal-audit-${runId.slice(0, 8)}.${format}`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) { setMessage((error as Error).message) }
  }

  return <section className="temporal-panel">
    <header className="temporal-header">
      <div><h2>Temporal QA <span>Thử nghiệm · Review từng điểm</span></h2><p>So sánh nhãn với chuyển động cùng track qua nhiều frame.</p></div>
      {summary && <div className="temporal-summary">
        <b>{Math.round(summary.coverage * 100)}% coverage</b><span>{summary.available}/{summary.total} điểm · {summary.warnings} cảnh báo</span><span>{summary.reviewed_temporal} Temporal · {summary.reviewed_human} Human · {summary.reviewed_total} đã review</span>
        <div><button className="btn-outline" onClick={() => downloadAudit('json')}>Audit JSON</button><button className="btn-outline" onClick={() => downloadAudit('csv')}>Audit CSV</button></div>
      </div>}
    </header>
    {message && <p className="notice">{message}</p>}
    <div className="temporal-filters">
      <label>Trạng thái <select value={status} onChange={event => setStatus(event.target.value)}><option value="all">Tất cả</option><option value="available">Có estimate</option><option value="unavailable">Không khả dụng</option></select></label>
      <label>Bằng chứng <select value={evidence} onChange={event => setEvidence(event.target.value)}><option value="all">Tất cả</option><option value="model_trajectory">Quỹ đạo model</option><option value="annotation_jitter">Nhãn bị giật</option><option value="face_r13">Face R13</option></select></label>
      <label>Review <select value={reviewFilter} onChange={event => setReviewFilter(event.target.value)}><option value="all">Tất cả</option><option value="unreviewed">Chưa review</option><option value="temporal">Dùng Temporal</option><option value="human">Giữ Human</option></select></label>
      {summary?.unavailable_reasons.map(item => <span className="temporal-reason" key={item.unavailable_reason}>{item.unavailable_reason}: {item.count}</span>)}
    </div>
    <div className="temporal-workspace">
      <aside className="temporal-list">
        {visibleItems.map(item => <button key={item.id} className={selected?.id === item.id ? 'active' : ''} onClick={() => loadDetail(item.id).catch(error => setMessage(error.message))}>
          <b>Frame {item.frame_index} · KP {item.keypoint_id}</b><small>{item.file_name}</small>
          <span>{item.evidence ? evidenceLabel[item.evidence] || item.evidence : item.estimate_status === 'available' ? item.method : item.unavailable_reason}</span>
          <em>Q {Math.round(item.temporal_quality * 100)}%{item.temporal_score != null ? ` · Score ${Math.round(item.temporal_score * 100)}%` : ''}</em>
          {reviewsByKey.get(reviewKey(item)) && <strong className={`temporal-review-badge ${reviewsByKey.get(reviewKey(item))!.source}`}>{reviewsByKey.get(reviewKey(item))!.source === 'temporal' ? 'Đã dùng Temporal' : 'Đã giữ Human'}</strong>}
        </button>)}
        {!visibleItems.length && <p>Không có điểm phù hợp bộ lọc.</p>}
        <div className="temporal-pagination">
          <button disabled={pageNumber <= 1} onClick={() => load(pageNumber - 1).catch(error => setMessage(error.message))}>← Trang trước</button>
          <span>Trang {pageNumber}/{totalPages} · {resultTotal} điểm</span>
          <button disabled={pageNumber >= totalPages} onClick={() => load(pageNumber + 1).catch(error => setMessage(error.message))}>Trang sau →</button>
        </div>
      </aside>
      <div className="temporal-inspector">
        {!selected ? <p>Chọn một kết quả temporal để xem chi tiết.</p> : <>
          <div className="temporal-nav"><button disabled={selectedIndex <= 0 && pageNumber <= 1} onClick={() => goPrevious().catch(error => setMessage(error.message))}>← Trước</button><b>Frame {selected.frame_index} · Track {selected.internal_track_id.slice(0, 8)}</b><button disabled={selectedIndex < 0 || (selectedIndex >= visibleItems.length - 1 && pageNumber >= totalPages)} onClick={() => goNext().catch(error => setMessage(error.message))}>Sau →</button></div>
          <div className="temporal-canvas">
            <img src={imageUrl(selected.file_name)} onLoad={event => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
            <svg viewBox={`0 0 ${size.width} ${size.height}`}>
              {trajectoryPoints && <polyline points={trajectoryPoints} className="temporal-path" />}
              {selected.human_x != null && selected.human_y != null && <g>
                <circle cx={selected.human_x} cy={selected.human_y} r={7 * overlayScale} className="temporal-human" />
                <text x={selected.human_x - 11 * overlayScale} y={selected.human_y - 10 * overlayScale} textAnchor="end" className="temporal-point-label human" style={{ fontSize: 18 * overlayScale, strokeWidth: 5 * overlayScale }}>{selected.keypoint_id}</text>
              </g>}
              {selected.temporal_x != null && selected.temporal_y != null && <g>
                <circle cx={selected.temporal_x} cy={selected.temporal_y} r={8 * overlayScale} className="temporal-estimate" />
                <text x={selected.temporal_x + 12 * overlayScale} y={selected.temporal_y - 10 * overlayScale} textAnchor="start" className="temporal-point-label estimate" style={{ fontSize: 18 * overlayScale, strokeWidth: 5 * overlayScale }}>{selected.keypoint_id}</text>
              </g>}
            </svg>
          </div>
          <div className="temporal-legend"><span className="human">Human · Point {selected.keypoint_id}</span><span className="estimate">Temporal · Point {selected.keypoint_id}</span></div>
          <div className="temporal-review-actions">
            <div>
              <b>{selectedReview ? (selectedReview.source === 'temporal' ? 'Đang dùng Temporal' : 'Đang giữ Human') : 'Chưa review'}</b>
              {selectedReview && <small>Tọa độ đã lưu: ({selectedReview.selected_x.toFixed(1)}, {selectedReview.selected_y.toFixed(1)})</small>}
            </div>
            <button className={selectedReview?.source === 'human' ? 'selected human' : ''} disabled={reviewBusy || !selected.annotation_id || selected.human_x == null || selected.human_y == null} onClick={() => saveReview('human')}>Giữ điểm Human</button>
            <button className={selectedReview?.source === 'temporal' ? 'selected temporal' : ''} title={!selected.evidence || selected.temporal_quality < 0.45 ? 'Temporal chưa có đủ bằng chứng tin cậy để thay điểm Human.' : undefined} disabled={reviewBusy || !selected.annotation_id || selected.estimate_status !== 'available' || selected.temporal_x == null || selected.temporal_y == null || !selected.evidence || selected.temporal_quality < 0.45} onClick={() => saveReview('temporal')}>Dùng điểm Temporal</button>
            <button className="clear" disabled={reviewBusy || !selectedReview} onClick={clearReview}>Bỏ lựa chọn</button>
          </div>
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
