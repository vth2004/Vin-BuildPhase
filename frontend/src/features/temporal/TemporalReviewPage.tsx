import { Link } from 'react-router-dom'
import TemporalPanel from './TemporalPanel'
import AiQcFramePanel from './AiQcFramePanel'
import TemporalRunOptions from './TemporalRunOptions'
import VideoImportPanel from '../video/VideoImportPanel'
import './temporal.css'
import { useTemporalWorkspace } from './useTemporalWorkspace'

export default function TemporalReviewPage({ feature }: { feature: 'pose' | 'face' }) {
  const { config, session, runs, datasets, datasetId, setDatasetId, windowSize, setWindowSize, temporalEnabled, setTemporalEnabled, busy, selectedRunId, setSelectedRunId, message, setMessage, zipFile, setZipFile, setAnnotationFile, setManifestFile, loadWorkspace, startRun, uploadZip } = useTemporalWorkspace(feature)

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
