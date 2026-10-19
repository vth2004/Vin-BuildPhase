import { readSession } from '../../shared/session'
import { FormEvent, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import TemporalPanel from './TemporalPanel'
import AiQcFramePanel from './AiQcFramePanel'
import TemporalRunOptions from './TemporalRunOptions'
import VideoImportPanel from '../video/VideoImportPanel'
import './temporal.css'

import { requestJson } from '../../shared/api'

import type { Dataset } from '../../shared/types'
type TemporalRun = {
  id: string
  dataset_id: string
  dataset_name: string
  status: string
  progress: number
  created_at: number
  temporal_enabled: boolean
  temporal_window_size: number
  input_mode: 'annotated' | 'ai_qc'
}

const CONFIG = {
  pose: { schemaId: 'vf_humanpose17_v1', title: 'Human Pose Temporal QA', backTo: '/pose', backLabel: 'Human Pose' },
  face: { schemaId: 'vf_face_landmark50_v1', title: 'Face Landmark Temporal QA', backTo: '/face', backLabel: 'Face Landmark' },
} as const

export default function TemporalReviewPage({ feature }: { feature: keyof typeof CONFIG }) {
  const config = CONFIG[feature]
  const session = readSession()
  const [searchParams, setSearchParams] = useSearchParams()
  const [runs, setRuns] = useState<TemporalRun[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [windowSize, setWindowSize] = useState(15)
  const [temporalEnabled, setTemporalEnabled] = useState(true)
  const [busy, setBusy] = useState(false)
  const [selectedRunId, setSelectedRunId] = useState(searchParams.get('run') || '')
  const [message, setMessage] = useState('')
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [manifestFile, setManifestFile] = useState<File | null>(null)

  const loadWorkspace = async (preferredDatasetId?: string) => {
    if (!session) return
    const [items, project] = await Promise.all([
      requestJson<TemporalRun[]>(`/projects/${session.projectId}/runs?schema_id=${config.schemaId}`, {}, session.recoveryKey),
      requestJson<{ datasets: Dataset[] }>(`/projects/${session.projectId}`, {}, session.recoveryKey),
    ])
    const temporalRuns = items.filter(run => run.temporal_enabled)
    const matchingDatasets = project.datasets.filter(dataset => dataset.schema_id === config.schemaId)
    setRuns(temporalRuns); setDatasets(matchingDatasets)
    setDatasetId(current => preferredDatasetId || current || matchingDatasets.find(dataset => dataset.status === 'ready')?.id || '')
    setSelectedRunId(current => current && temporalRuns.some(run => run.id === current) ? current : temporalRuns.find(run => run.status === 'completed')?.id || temporalRuns[0]?.id || '')
  }

  useEffect(() => {
    if (!session) return
    loadWorkspace().catch(error => setMessage(error.message))
  }, [session?.projectId, config.schemaId])

  useEffect(() => {
    if (selectedRunId) setSearchParams({ run: selectedRunId }, { replace: true })
  }, [selectedRunId, setSearchParams])

  const startRun = async () => {
    if (!session || !datasetId || !temporalEnabled) return
    setBusy(true); setMessage('Đang tạo Temporal run...')
    try {
      const created = await requestJson<{ id: string }>(`/projects/${session.projectId}/runs?dataset_id=${datasetId}&temporal_enabled=true&window_size=${windowSize}`, {
        method: 'POST',
      }, session.recoveryKey)
      setSelectedRunId(created.id)
      const poll = async (): Promise<void> => {
        const state = await requestJson<{ status: string; progress: number; error_message?: string; input_mode?: 'annotated' | 'ai_qc' }>(`/projects/${session.projectId}/runs/${created.id}`, {}, session.recoveryKey)
        setRuns(current => [{ id: created.id, dataset_id: datasetId, dataset_name: datasets.find(item => item.id === datasetId)?.name || datasetId, status: state.status, progress: state.progress, created_at: Date.now(), temporal_enabled: true, temporal_window_size: windowSize, input_mode: state.input_mode || 'annotated' }, ...current.filter(run => run.id !== created.id)])
        setMessage(`Temporal run: ${state.status} (${state.progress}%)`)
        if (state.status === 'completed') { setBusy(false); return }
        if (state.status === 'failed') { setBusy(false); throw new Error(state.error_message || 'Temporal run thất bại.') }
        await new Promise(resolve => window.setTimeout(resolve, 1000))
        await poll()
      }
      await poll()
    } catch (error) { setBusy(false); setMessage((error as Error).message) }
  }

  const uploadZip = async (event: FormEvent) => {
    event.preventDefault()
    if (!session || !zipFile) return
    setBusy(true); setMessage('Đang tải dataset lên...')
    try {
      const body = new FormData(); body.append('file', zipFile); body.append('schema_id', config.schemaId)
      if (annotationFile) body.append('annotation_file', annotationFile)
      if (manifestFile) body.append('sequence_manifest_file', manifestFile)
      const dataset = await requestJson<Dataset>(`/projects/${session.projectId}/datasets`, { method: 'POST', body }, session.recoveryKey)
      setZipFile(null); setAnnotationFile(null); setManifestFile(null)
      await loadWorkspace(dataset.id)
      setMessage(`Đã nhập ${dataset.image_count} frame vào ${dataset.name} và tự chọn dataset này.`)
    } catch (error) { setMessage((error as Error).message) } finally { setBusy(false) }
  }

  if (!session) return <main className="tool-page temporal-review-page">
    <header><div><h1>{config.title}</h1><small>Chưa có project đang mở.</small></div></header>
    <section className="temporal-page-empty"><p>Hãy mở project trước khi review kết quả Temporal.</p><Link className="btn-outline" to="/pose">Mở project</Link></section>
  </main>

  const selectedRun = runs.find(run => run.id === selectedRunId)
  return <main className="tool-page temporal-review-page">
    <header>
      <div><h1>{config.title}</h1><small>{session.projectName} · Màn review Temporal riêng</small></div>
      <Link className="btn-outline" to={config.backTo}>← Quay lại {config.backLabel}</Link>
    </header>
    {message && <p className="notice">{message}</p>}
    <section className="temporal-import">
      <div><h2>1. Nhập dữ liệu Temporal</h2><p>Upload trực tiếp tại đây; annotation là tùy chọn. Dataset mới sẽ được tự động chọn để cấu hình run.</p></div>
      <form onSubmit={uploadZip} className="temporal-upload-form">
        <label>ZIP frame<input type="file" accept=".zip,application/zip" onChange={event => setZipFile(event.target.files?.[0] || null)} /></label>
        <label>Annotation tùy chọn<input type="file" accept={feature === 'face' ? '.xml,.json,application/xml,application/json' : '.json,application/json'} onChange={event => setAnnotationFile(event.target.files?.[0] || null)} /></label>
        <label>Sequence manifest tùy chọn<input type="file" accept=".json,application/json" onChange={event => setManifestFile(event.target.files?.[0] || null)} /></label>
        <button disabled={!zipFile || busy}>Upload ZIP</button>
      </form>
      <VideoImportPanel session={session} schemaId={config.schemaId} annotationAccept={feature === 'face' ? '.xml,.json,application/xml,application/json' : '.json,application/json'} existingDatasetId={datasets.find(item => item.status !== 'ready')?.id} onMessage={setMessage} onReady={async readyId => { await loadWorkspace(readyId) }} />
    </section>
    <section className="temporal-create-run">
      <div><h2>2. Cấu hình và chạy Temporal</h2><p>Chọn dataset, xác nhận sequence mapping và cấu hình cửa sổ tại đây.</p></div>
      <label>Dataset
        <select value={datasetId} onChange={event => setDatasetId(event.target.value)}>
          <option value="">Chọn dataset</option>
          {datasets.map(dataset => <option key={dataset.id} value={dataset.id} disabled={dataset.status !== 'ready'}>{dataset.name} ({dataset.image_count} ảnh · {dataset.status})</option>)}
        </select>
      </label>
      {datasetId && datasets.find(dataset => dataset.id === datasetId)?.status === 'ready' && <TemporalRunOptions session={session} datasetId={datasetId} enabled={temporalEnabled} windowSize={windowSize} maxWindowSize={feature === 'face' ? 69 : 29} onEnabled={setTemporalEnabled} onWindow={setWindowSize} />}
      <button disabled={!datasetId || !temporalEnabled || busy || datasets.find(dataset => dataset.id === datasetId)?.status !== 'ready'} onClick={startRun}>Chạy Temporal QA</button>
    </section>
    <section className="temporal-run-picker">
      <div><h2>3. Chọn Temporal run để review</h2><p>Chỉ các run đã bật Temporal QA mới xuất hiện ở đây.</p></div>
      <label>Run
        <select value={selectedRunId} onChange={event => setSelectedRunId(event.target.value)}>
          <option value="">Chọn run</option>
          {runs.map(run => <option key={run.id} value={run.id} disabled={run.status !== 'completed'}>
            {run.dataset_name} · {run.status} · cửa sổ {run.temporal_window_size} · {run.id.slice(0, 8)}
          </option>)}
        </select>
      </label>
    </section>
    {!runs.length && <section className="temporal-page-empty"><p>Chưa có run Temporal. Quay lại màn {config.backLabel}, bật Temporal QA rồi tạo run.</p></section>}
    {selectedRun && selectedRun.status !== 'completed' && <section className="temporal-page-empty"><p>Run đang ở trạng thái <b>{selectedRun.status}</b> ({selectedRun.progress}%). Hãy đợi run hoàn thành.</p></section>}
    {selectedRun?.status === 'completed' && (selectedRun.input_mode === 'ai_qc'
      ? <AiQcFramePanel session={session} runId={selectedRun.id} datasetId={selectedRun.dataset_id} feature={feature} />
      : <TemporalPanel session={session} runId={selectedRun.id} datasetId={selectedRun.dataset_id} />)}
  </main>
}
