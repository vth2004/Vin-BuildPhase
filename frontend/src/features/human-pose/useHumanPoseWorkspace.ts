import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { API, requestJson } from '../../shared/api'
import { readSession, saveSession } from '../../shared/session'
import type { Dataset, Session } from '../../shared/types'
import type { Warning, ImageDetails } from './types'

export function useHumanPoseWorkspace() {
  const [session, setSession] = useState<Session | null>(readSession)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const schemaId = 'vf_humanpose17_v1'
  const [runId, setRunId] = useState<string | null>(null)
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<Warning[]>([])
  const [selectedWarning, setSelectedWarning] = useState<Warning | null>(null)
  const [message, setMessage] = useState('')
  const [imageDetails, setImageDetails] = useState<ImageDetails | null>(null)
  const [runMode, setRunMode] = useState<string | null>(null)
  const [aiDraftRunId, setAiDraftRunId] = useState<string | null>(null)

  // Bộ lọc
  const [minSuspicion, setMinSuspicion] = useState<number>(0.2)
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'reviewed'>('all')
  const [jointFilter, setJointFilter] = useState<string>('all')
  const [searchImage, setSearchImage] = useState<string>('')
  const [reliabilityFilter, setReliabilityFilter] = useState<'all' | 'high' | 'hard'>('all')


  const headers = () => ({ Authorization: `Bearer ${session?.recoveryKey}` })

  const request = (path: string, init: RequestInit = {}) =>
    requestJson<any>(path, init, session?.recoveryKey, 'Request failed')

  const loadProject = async () => {
    if (!session) return
    const data = await request(`/projects/${session.projectId}`)
    const poseDatasets = (data.datasets || []).filter(
      (d: Dataset) => d.schema_id === schemaId || d.schema_id === 'vf_humanpose17_v1'
    )
    setDatasets(poseDatasets)
    if (poseDatasets.length > 0) {
      setActiveDatasetId(curr => (poseDatasets.some((d: Dataset) => d.id === curr) ? curr : poseDatasets[0].id))
    } else {
      setActiveDatasetId(null)
    }
  }

  useEffect(() => {
    loadProject().catch(e => setMessage(e.message))
  }, [session?.projectId])

  useEffect(() => {
    if (!runId || !session) return
    let timerId: any
    const tick = async () => {
      const run = await request(`/projects/${session.projectId}/runs/${runId}`)
      if (run.status === 'completed' && run.input_mode === 'ai_qc') setAiDraftRunId(runId)
      if (run.mode) setRunMode(run.mode)
      const modeLabel = run.mode === 'K2' ? 'Ensemble K=2' : run.mode === 'K1' || run.mode === 'K1_fast' ? 'K=1 Nhanh' : run.mode === 'K1_fallback' ? 'K=1 (Fallback)' : run.mode ? run.mode : ''
      setMessage(`Tiến độ Run: ${run.status} (${run.progress}%) ${modeLabel ? `[${modeLabel}]` : ''}`)
      if (run.status === 'completed' || run.status === 'failed') {
        clearInterval(timerId)
        if (run.status === 'completed') {
          const items: Warning[] = await request(`/projects/${session.projectId}/runs/${runId}/warnings`)
          setWarnings(items)
          setSelectedWarning(prev => prev ?? (items.length > 0 ? items[0] : null))
        }
      }
    }
    tick().catch(e => setMessage(e.message))
    timerId = setInterval(tick, 1000)
    return () => clearInterval(timerId)
  }, [runId, session?.projectId])


  // Lấy chi tiết khung xương khi chọn cảnh báo
  useEffect(() => {
    if (!selectedWarning || !runId || !session) return
    const fetchKeypoints = async () => {
      try {
        const data = await request(
          `/projects/${session.projectId}/runs/${runId}/warnings/${selectedWarning.id}/keypoints`
        )
        setImageDetails(data)
      } catch {
        setImageDetails(null)
      }
    }
    fetchKeypoints()
  }, [selectedWarning?.id, runId, session?.projectId])

  // Danh sách các loại khớp duy nhất để làm bộ lọc
  const uniqueJoints = useMemo(() => {
    const s = new Set<string>()
    warnings.forEach(w => s.add(w.keypoint))
    return Array.from(s).sort()
  }, [warnings])

  // Danh sách cảnh báo sau khi lọc
  const filteredWarnings = useMemo(() => {
    return warnings.filter(w => {
      if (w.suspicion < minSuspicion) return false
      if (statusFilter === 'pending' && w.review !== null) return false
      if (statusFilter === 'reviewed' && w.review === null) return false
      if (jointFilter !== 'all' && w.keypoint !== jointFilter) return false
      if (searchImage && !w.image_name.toLowerCase().includes(searchImage.toLowerCase())) return false
      if (reliabilityFilter === 'high' && (w.reliability ?? 1.0) < 0.70) return false
      if (reliabilityFilter === 'hard' && (w.reliability ?? 1.0) >= 0.50 && w.warning_type !== 'hard_case') return false
      return true
    })
  }, [warnings, minSuspicion, statusFilter, jointFilter, searchImage, reliabilityFilter])

  const createProject = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const name = new FormData(event.currentTarget).get('name')
    try {
      const data = await request('/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
      const next = { projectId: data.id, recoveryKey: data.recovery_key, projectName: data.name }
      saveSession(next)
      setSession(next)
      setMessage('Lưu Recovery key của bạn để sử dụng khi quay lại.')
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const resumeProject = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const pId = (form.get('projectId') as string)?.trim()
    const rKey = (form.get('recoveryKey') as string)?.trim()
    try {
      const res = await fetch(`${API}/projects/${pId}`, {
        headers: { Authorization: `Bearer ${rKey}` },
      })
      if (!res.ok) throw new Error('Recovery key hoặc Project ID không hợp lệ')
      const data = await res.json()
      const next = { projectId: pId, recoveryKey: rKey, projectName: data.project.name }
      saveSession(next)
      setSession(next)
      setMessage(`Đã mở lại project: ${data.project.name}`)
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const handleZipSelect = (f: File | null) => {
    setFile(f)
    setValidationError(null)
    if (!f) return
    const name = f.name.toLowerCase()
    const faceTokens = ['face', 'vf50', 'landmark50', 'landmark_50', 'nguquan', 'part_nghia', 'part_tung']
    if (faceTokens.some(t => name.includes(t))) {
      setValidationError('⚠️ Cảnh báo: Tên tệp ZIP có dấu hiệu là dữ liệu Face Landmark (VF50). Phân hệ Human Pose chỉ nhận dữ liệu Pose toàn thân (COCO 17 điểm). Vui lòng kiểm tra lại trước khi tải lên.')
    }
  }

  const handleAnnotationSelect = (f: File | null) => {
    setAnnotationFile(f)
    setValidationError(null)
    if (!f) return
    const name = f.name.toLowerCase()
    if (name.endsWith('.xml')) {
      setValidationError('⚠️ File XML là định dạng của Face Landmark (CVAT XML). Phân hệ Human Pose chỉ hỗ trợ file JSON COCO 17 điểm. Vui lòng chuyển sang phân hệ Face Landmark.')
      return
    }
    if (name.includes('face') || name.includes('vf50') || name.includes('landmark50')) {
      setValidationError('⚠️ File nhãn có tên Face Landmark (VF50). Vui lòng kiểm tra lại hoặc chuyển sang phân hệ Face Landmark.')
      return
    }
    if (name.endsWith('.json')) {
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const json = JSON.parse(reader.result as string)
          if (json.schema_id === 'vf_face_landmark50_v1') {
            setValidationError('⚠️ File nhãn này thuộc schema Face Landmark (vf_face_landmark50_v1). Vui lòng chuyển sang phân hệ Face Landmark để kiểm định.')
          } else if (Array.isArray(json.categories)) {
            const catNames = json.categories.map((c: any) => String(c.name || '').toLowerCase())
            const faceHints = ['longmaytrai', 'longmayphai', 'songmui', 'mattrai', 'matphai', 'moingoai', 'moitrong', 'face', 'vf50', 'landmark50']
            const found = catNames.filter((c: string) => faceHints.includes(c))
            if (found.length > 0) {
              setValidationError(`⚠️ File nhãn này chứa các danh mục Face Landmark (${found.join(', ')}). Phân hệ Human Pose chỉ nhận dữ liệu Pose toàn thân (COCO 17 điểm).`)
            }
          }
        } catch {
          // Bỏ qua lỗi cú pháp đọc nhanh
        }
      }
      reader.readAsText(f)
    }
  }

  const upload = async () => {
    if (!file || !session) return
    if (validationError && validationError.startsWith('⚠️ File')) {
      setMessage(validationError)
      return
    }
    try {
      const body = new FormData()
      body.append('file', file)
      if (annotationFile) {
        body.append('annotation_file', annotationFile)
      }
      body.append('schema_id', schemaId)
      const data = await request(`/projects/${session.projectId}/datasets`, { method: 'POST', body })
      setFile(null)
      setAnnotationFile(null)
      setValidationError(null)
      setActiveDatasetId(data.id)
      setMessage(`Đã nhận dataset ${data.name}: ${data.image_count} ảnh theo ${data.schema_id}.`)
      loadProject()
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const startRun = async (datasetId: string) => {
    const targetDataset = datasets.find(d => d.id === datasetId)
    if (targetDataset && targetDataset.schema_id !== schemaId && targetDataset.schema_id !== 'vf_humanpose17_v1') {
      setMessage(`Không thể chạy YOLO26s-pose trên dataset '${targetDataset.name}' vì schema là ${targetDataset.schema_id} (chỉ hỗ trợ vf_humanpose17_v1).`)
      return
    }
    try {
      setActiveDatasetId(datasetId)
      const data = await request(`/projects/${session!.projectId}/runs?dataset_id=${datasetId}`, { method: 'POST' })
      setRunId(data.id)
      setAiDraftRunId(null)
      setWarnings([])
      setSelectedWarning(null)
      setMessage(`Đã bắt đầu chạy Run (${data.mode === 'real' ? 'Model YOLO26s-pose thật' : data.mode || 'Ensemble'})...`)
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const downloadAiDraft = async () => {
    if (!session || !aiDraftRunId) return
    try {
      const response = await fetch(`${API}/projects/${session.projectId}/runs/${aiDraftRunId}/ai-drafts/export`, { headers: headers() })
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || 'Không thể export AI draft')
      const url = URL.createObjectURL(await response.blob())
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `pose_ai_draft_${aiDraftRunId}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  const review = async (warning: Warning, value: string) => {
    try {
      await request(`/projects/${session!.projectId}/warnings/${warning.id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: value }),
      })
      setWarnings(items => items.map(x => (x.id === warning.id ? { ...x, review: value } : x)))
      if (selectedWarning?.id === warning.id) {
        setSelectedWarning({ ...selectedWarning, review: value })
      }
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Bỏ qua nếu đang gõ trong input text
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return
      if (!selectedWarning) return

      const key = e.key.toLowerCase()

      if (key === 'k') {
        review(selectedWarning, 'keep')
      } else if (key === 'm') {
        review(selectedWarning, 'use_suggestion')
      } else if (key === 's') {
        review(selectedWarning, 'skip')
      } else if (e.key === 'ArrowDown' || key === 'j') {
        e.preventDefault()
        const currIdx = filteredWarnings.findIndex(w => w.id === selectedWarning.id)
        if (currIdx >= 0 && currIdx < filteredWarnings.length - 1) {
          setSelectedWarning(filteredWarnings[currIdx + 1])
        }
      } else if (e.key === 'ArrowUp' || key === 'u') {
        e.preventDefault()
        const currIdx = filteredWarnings.findIndex(w => w.id === selectedWarning.id)
        if (currIdx > 0) {
          setSelectedWarning(filteredWarnings[currIdx - 1])
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedWarning, filteredWarnings])

  const exportCleaned = async () => {
    if (!runId || !session) return
    try {
      const data = await request(`/projects/${session.projectId}/runs/${runId}/export`)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `cleaned_annotation_${runId.slice(0, 8)}.json`
      a.click()
      URL.revokeObjectURL(url)
      setMessage('Đã tải xuống file annotation đã cập nhật thành công!')
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const imageUrl = session && selectedWarning && activeDatasetId
    ? `${API}/projects/${session.projectId}/datasets/${activeDatasetId}/images/${selectedWarning.image_name}?token=${session.recoveryKey}`
    : null

  return { session, setSession, datasets, file, validationError, setActiveDatasetId, warnings, selectedWarning, setSelectedWarning, message, imageDetails, runMode, aiDraftRunId, minSuspicion, setMinSuspicion, statusFilter, setStatusFilter, jointFilter, setJointFilter, searchImage, setSearchImage, reliabilityFilter, setReliabilityFilter, uniqueJoints, filteredWarnings, createProject, resumeProject, handleZipSelect, handleAnnotationSelect, upload, startRun, downloadAiDraft, review, exportCleaned, imageUrl, }
}
