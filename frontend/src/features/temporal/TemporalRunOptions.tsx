import { useEffect, useMemo, useState } from 'react'
import './temporal.css'

import { API, requestJson } from '../../shared/api'
import type { ProjectCredentials as Session } from '../../shared/types'
type Frame = { file_name: string; original_frame_index: number | null }
type Sequence = { id: string; name: string; frame_count: number; frame_order_source: string; frame_deltas: number[]; confirmed_at: number | null; fps: number | null; requires_manifest: boolean; reason?: string | null; frames: Frame[] }
type Summary = { temporal_ready: boolean; sequences: Sequence[] }
type Props = { session: Session; datasetId: string; enabled: boolean; windowSize: number; maxWindowSize?: number; onEnabled: (value: boolean) => void; onWindow: (value: number) => void }

export default function TemporalRunOptions({ session, datasetId, enabled, windowSize, maxWindowSize = 29, onEnabled, onWindow }: Props) {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [fps, setFps] = useState('')
  const [message, setMessage] = useState('')
  const load = async () => {
    if (!datasetId) return setSummary(null)
    setSummary(await requestJson<Summary>(`/projects/${session.projectId}/datasets/${datasetId}/sequences`, {}, session.recoveryKey, 'Không đọc được metadata sequence'))
  }
  useEffect(() => { load().catch(error => setMessage(error.message)) }, [datasetId])
  const confirm = async () => {
    const parsedFps = fps ? Number(fps) : null
    const next = await requestJson<Summary>(`/projects/${session.projectId}/datasets/${datasetId}/sequences/confirm`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fps: parsedFps, sampling_confirmed: parsedFps !== null, confirmation_source: 'ui_phase5' }),
    }, session.recoveryKey, 'Xác nhận sequence thất bại')
    setSummary(next); setMessage('Đã lưu xác nhận sequence và snapshot mapping.')
  }
  const duration = useMemo(() => { const known = summary?.sequences.find(item => item.fps)?.fps; return known ? `${(windowSize / known).toFixed(2)} giây` : null }, [summary, windowSize])
  const imageUrl = (name: string) => `${API}/projects/${session.projectId}/datasets/${datasetId}/images/${name.split('/').map(encodeURIComponent).join('/')}?token=${encodeURIComponent(session.recoveryKey)}`

  return <div className="temporal-options">
    <label className="temporal-toggle"><input type="checkbox" checked={enabled} disabled={!summary?.temporal_ready} onChange={event => onEnabled(event.target.checked)} /><span><b>Temporal QA</b> <em>Thử nghiệm</em><small>Phân tích offline bằng frame trước và sau; QA/QC hiện tại không thay đổi.</small></span></label>
    <label>Cửa sổ <select value={windowSize} disabled={!enabled} onChange={event => onWindow(Number(event.target.value))}>{Array.from({ length: Math.floor((maxWindowSize - 5) / 2) + 1 }, (_, index) => 5 + index * 2).map(value => <option key={value}>{value}</option>)}</select> frame {duration && <small>≈ {duration}</small>}</label>
    {message && <p className="notice">{message}</p>}
    {summary && !summary.temporal_ready && <p className="temporal-help">Temporal chưa bật được: xác nhận mapping hoặc cung cấp manifest nếu mapping mơ hồ.</p>}
    <div className="sequence-grid">{summary?.sequences.map(sequence => {
      const gapIndex = sequence.frame_deltas.findIndex(delta => delta !== sequence.frame_deltas[0])
      const previews = [sequence.frames[0], gapIndex >= 0 ? sequence.frames[gapIndex] : undefined, gapIndex >= 0 ? sequence.frames[gapIndex + 1] : undefined, sequence.frames.at(-1)].filter((frame, index, all) => frame && all.indexOf(frame) === index) as Frame[]
      return <article className="sequence-card" key={sequence.id}><div><b>{sequence.name}</b><span>{sequence.frame_count} frame · {sequence.frame_order_source}</span></div><p>Delta thực: {[...new Set(sequence.frame_deltas)].join(', ') || '—'} · {sequence.confirmed_at ? 'Đã xác nhận' : 'Chưa xác nhận'}</p>{sequence.reason && <p>{sequence.reason}</p>}<div className="sequence-thumbs">{previews.map(frame => <figure key={frame.file_name}><img src={imageUrl(frame.file_name)} loading="lazy" /><figcaption>{frame.original_frame_index ?? '?'}</figcaption></figure>)}</div>{sequence.requires_manifest && <strong className="temporal-error">Mapping mơ hồ — cần sequence_manifest.json.</strong>}</article>
    })}</div>
    {summary && !summary.temporal_ready && !summary.sequences.some(item => item.requires_manifest) && <div className="sequence-confirm"><label>FPS đã lấy mẫu (tùy chọn)<input type="number" min="0.01" max="1000" step="0.01" value={fps} onChange={event => setFps(event.target.value)} /></label><button type="button" onClick={() => confirm().catch(error => setMessage(error.message))}>Xác nhận mapping</button></div>}
  </div>
}
