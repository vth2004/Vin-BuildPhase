import { useRunMonitor } from '../../shared/useRunMonitor'
import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { requestJson } from '../../shared/api'
import { readSession } from '../../shared/session'
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

export function useTemporalWorkspace(feature: 'pose' | 'face') {
  const config = CONFIG[feature]
  const session = readSession()
  const { monitor, cancel } = useRunMonitor(session)
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
      await monitor(created.id, state => {
        setRuns(current => [{ id: created.id, dataset_id: datasetId, dataset_name: datasets.find(item => item.id === datasetId)?.name || datasetId, status: state.status, progress: state.progress, created_at: Date.now(), temporal_enabled: true, temporal_window_size: windowSize, input_mode: state.input_mode || 'annotated' }, ...current.filter(run => run.id !== created.id)])
        setMessage(`Temporal run: ${state.status} (${state.progress}%)`)
        if (state.status === 'completed') { setBusy(false); return }
        if (state.status === 'failed') { setBusy(false); throw new Error(state.error_message || 'Temporal run thất bại.') }
      })
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

  return { config, session, runs, datasets, datasetId, setDatasetId, windowSize, setWindowSize, temporalEnabled, setTemporalEnabled, busy, selectedRunId, setSelectedRunId, message, setMessage, zipFile, setZipFile, setAnnotationFile, setManifestFile, loadWorkspace, startRun, uploadZip }
}
