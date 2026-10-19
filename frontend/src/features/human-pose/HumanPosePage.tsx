import { readSession, saveSession, clearSession } from '../../shared/session'
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'

import { API, requestJson } from '../../shared/api'

import type { Dataset, Session } from '../../shared/types'

type Warning = {
  id: string
  run_id: string
  image_name: string
  keypoint: string
  warning_type: string
  suspicion: number
  human_x: number
  human_y: number
  suggested_x: number
  suggested_y: number
  review: string | null
  reliability?: number
  delta?: number
}

type KeypointDetail = {
  id: number
  name?: string
  x: number | null
  y: number | null
  state: string
}

type ImageDetails = {
  keypoints: KeypointDetail[]
  edges: [number, number][]
  width: number
  height: number
}

const KEYPOINT_SIGMA_MAP: Record<string, { sigma: number; label: string }> = {
  nose: { sigma: 0.026, label: 'Rất hẹp' },
  r_eye: { sigma: 0.025, label: 'Rất hẹp' },
  l_eye: { sigma: 0.025, label: 'Rất hẹp' },
  r_ear: { sigma: 0.035, label: 'Hẹp' },
  l_ear: { sigma: 0.035, label: 'Hẹp' },
  r_shoulder: { sigma: 0.079, label: 'Rộng' },
  l_shoulder: { sigma: 0.079, label: 'Rộng' },
  r_elbow: { sigma: 0.072, label: 'Trung bình' },
  l_elbow: { sigma: 0.072, label: 'Trung bình' },
  r_wrist: { sigma: 0.062, label: 'Trung bình' },
  l_wrist: { sigma: 0.062, label: 'Trung bình' },
  r_hip: { sigma: 0.107, label: 'Rất rộng' },
  l_hip: { sigma: 0.107, label: 'Rất rộng' },
  r_knee: { sigma: 0.087, label: 'Rộng' },
  l_knee: { sigma: 0.087, label: 'Rộng' },
  r_ankle: { sigma: 0.089, label: 'Rộng' },
  l_ankle: { sigma: 0.089, label: 'Rộng' },
}

export default function HumanPosePage() {
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
  const [imgNaturalSize, setImgNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [imageDetails, setImageDetails] = useState<ImageDetails | null>(null)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [runMode, setRunMode] = useState<string | null>(null)
  const [aiDraftRunId, setAiDraftRunId] = useState<string | null>(null)

  // Bộ lọc
  const [minSuspicion, setMinSuspicion] = useState<number>(0.2)
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'reviewed'>('all')
  const [jointFilter, setJointFilter] = useState<string>('all')
  const [searchImage, setSearchImage] = useState<string>('')
  const [reliabilityFilter, setReliabilityFilter] = useState<'all' | 'high' | 'hard'>('all')

  const imgRef = useRef<HTMLImageElement>(null)

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

  const renderBadge = (type: string) => {
    if (type === 'swap_error') return <span className="badge badge-swap">Hoán đổi Trái-Phải</span>
    if (type === 'hard_case') return <span className="badge badge-hard">Điểm khó (bị che)</span>
    return <span className="badge badge-error">Nghi sai</span>
  }

  const imageUrl = session && selectedWarning && activeDatasetId
    ? `${API}/projects/${session.projectId}/datasets/${activeDatasetId}/images/${selectedWarning.image_name}?token=${session.recoveryKey}`
    : null

  // Tạo map id -> keypoint để vẽ skeleton. Hook phải chạy trước mọi early return.
  const kpMap = useMemo(() => {
    const map: Record<number, KeypointDetail> = {}
    if (imageDetails?.keypoints) {
      imageDetails.keypoints.forEach(k => {
        map[k.id] = k
      })
    }
    return map
  }, [imageDetails])

  if (!session) {
    return (
      <main className="tool-page pose-page">
        <header>
          <div>
            <h1>Khởi tạo & Quản lý Project</h1>
            <small>Tạo dự án mới hoặc nhập mã khôi phục để tiếp tục phiên làm việc kiểm định.</small>
          </div>
        </header>

        {message && <p className="notice">{message}</p>}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginTop: '16px' }}>
          <section style={{ margin: 0 }}>
            <h2>Tạo dự án mới</h2>
            <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', marginTop: 0 }}>
              Khởi tạo không gian làm việc riêng cho tập dữ liệu Pose/Landmark mới.
            </p>
            <form onSubmit={createProject} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <input name="name" required placeholder="Ví dụ: VinFast VF3 HumanPose QA" style={{ margin: 0, width: '100%' }} />
              <button style={{ width: '100%' }}>Tạo project mới</button>
            </form>
          </section>

          <section style={{ margin: 0 }}>
            <h2>Mở lại dự án đã có</h2>
            <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', marginTop: 0 }}>
              Tiếp tục phiên làm việc với Project ID và Recovery Key đã được cấp.
            </p>
            <form onSubmit={resumeProject} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <input name="projectId" required placeholder="Project ID (UUID)" style={{ margin: 0, width: '100%' }} />
              <input name="recoveryKey" required placeholder="Recovery Key" style={{ margin: 0, width: '100%' }} />
              <button className="btn-outline" style={{ width: '100%' }}>Mở lại project</button>
            </form>
          </section>
        </div>
      </main>
    )
  }

  return (
    <main className="tool-page pose-page">
      <header>
        <div>
          <h1>{session.projectName}</h1>
          <small>
            Project ID: <code>{session.projectId}</code> | Recovery key: <code>{session.recoveryKey}</code>
          </small>
        </div>
        <button
          className="btn-outline"
          onClick={() => {
            clearSession()
            setSession(null)
          }}
        >
          Rời project
        </button>
      </header>

      {message && <p className="notice">{message}</p>}
      {aiDraftRunId && <p><button className="btn-export" onClick={downloadAiDraft}>Tải nhãn AI draft (landmark-qa/v1)</button></p>}

      {/* 1. Upload Dataset */}
      <section>
        <h2>1. Upload Dataset đã gán nhãn</h2>
        <p style={{ fontSize: '0.88rem', color: '#64748b', marginTop: 0 }}>
          Hỗ trợ file ZIP xuất trực tiếp từ CVAT (gồm cả ảnh & nhãn COCO 17 điểm), hoặc ZIP ảnh kèm file JSON nhãn riêng.
        </p>
        <p><b>Schema bắt buộc:</b> VinFast HumanPose-17 (COCO 17 điểm toàn thân)</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>
            <span>Tệp ZIP ảnh (hoặc ảnh + nhãn):</span>
            <input type="file" accept=".zip,application/zip" onChange={e => handleZipSelect(e.target.files?.[0] || null)} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>
            <span>Tệp JSON nhãn tùy chọn (COCO 1.0):</span>
            <input type="file" accept=".json,application/json" onChange={e => handleAnnotationSelect(e.target.files?.[0] || null)} />
          </label>
        </div>

        {validationError && (
          <div style={{ marginTop: 12, padding: '10px 14px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, color: '#b91c1c', fontSize: '0.84rem', fontWeight: 600 }}>
            {validationError}
          </div>
        )}

        <div style={{ marginTop: 12 }}>
          <button disabled={!file} onClick={upload}>
            Tải lên dataset
          </button>
        </div>
      </section>

      {/* 2. Quản lý Dataset & Khởi chạy Run */}
      <section>
        <h2>2. Danh sách Dataset & Quét lỗi với Model AI</h2>
        <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: 0 }}>
          Chỉ hiển thị các dataset thuộc schema <b>VinFast HumanPose-17 (COCO 17 điểm)</b>. Các tập dữ liệu Face Landmark sẽ hiển thị ở phân hệ Face Landmark.
        </p>
        {datasets.length === 0 ? (
          <p style={{ color: '#64748b' }}>Chưa có dataset Human Pose nào được tải lên trong project này.</p>
        ) : (
          <div className="dataset-list">
            {datasets.map(d => (
              <div className="dataset-item" key={d.id}>
                <span>
                  <strong>{d.name}</strong> · {d.image_count} ảnh · Schema: <code>{d.schema_id}</code>
                  {d.status !== 'ready' && <> · <b>{d.status}</b></>}
                </span>
                <button className="btn-outline" disabled={d.status !== 'ready'} onClick={() => setActiveDatasetId(d.id)}>Cấu hình</button>
                <button disabled={d.status !== 'ready'} onClick={() => startRun(d.id)}>
                  Chạy quét lỗi (YOLO26s-pose)
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* 3. Review trực quan: Bộ lọc + Danh sách cảnh báo + Canvas Skeleton */}
      {warnings.length > 0 && (
        <section>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
            <div>
              <h2 style={{ marginBottom: 4 }}>
                3. Review trực quan ({filteredWarnings.length}/{warnings.length} điểm nghi ngờ)
                {runMode && (
                  <span className="badge badge-mode" style={{ marginLeft: 8, fontSize: '0.78rem', verticalAlign: 'middle' }}>
                    {runMode === 'K2' ? 'Ensemble K=2 (YOLO26s + RTMPose)' : runMode === 'K1' || runMode === 'K1_fast' ? 'K=1 Nhanh' : runMode === 'K1_fallback' ? 'K=1 (Fallback)' : `Mode: ${runMode}`}
                  </span>
                )}
              </h2>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Phím tắt: <kbd>K</kbd> Giữ nhãn | <kbd>M</kbd> Nhận Model | <kbd>S</kbd> Bỏ qua | <kbd>↓</kbd>/<kbd>↑</kbd> Chuyển mục
              </span>
            </div>
            <button className="btn-export" onClick={exportCleaned}>
              📥 Xuất File Annotation đã sửa (JSON)
            </button>
          </div>

          {/* Thanh bộ lọc (Filters Bar) */}
          <div className="filters-bar">
            <div className="filter-group">
              <span>Độ nghi ngờ:</span>
              <button
                className={`btn-outline ${minSuspicion === 0.2 ? 'active' : ''}`}
                onClick={() => setMinSuspicion(0.2)}
              >
                Tất cả (≥20%)
              </button>
              <button
                className={`btn-outline ${minSuspicion === 0.6 ? 'active' : ''}`}
                onClick={() => setMinSuspicion(0.6)}
              >
                Đáng ngờ (≥60%)
              </button>
              <button
                className={`btn-outline ${minSuspicion === 0.8 ? 'active' : ''}`}
                onClick={() => setMinSuspicion(0.8)}
              >
                Nghiêm trọng (≥80%)
              </button>
            </div>

            <div className="filter-group">
              <span>Độ tin cậy:</span>
              <select value={reliabilityFilter} onChange={e => setReliabilityFilter(e.target.value as any)}>
                <option value="all">Tất cả</option>
                <option value="high">Tin cậy cao (R ≥ 70%)</option>
                <option value="hard">Ca khó / Bất định (R &lt; 50%)</option>
              </select>
            </div>

            <div className="filter-group">
              <span>Trạng thái:</span>
              <select value={statusFilter} onChange={e => setStatusFilter(e.target.value as any)}>
                <option value="all">Tất cả</option>
                <option value="pending">Chưa review</option>
                <option value="reviewed">Đã review</option>
              </select>
            </div>

            <div className="filter-group">
              <span>Khớp:</span>
              <select value={jointFilter} onChange={e => setJointFilter(e.target.value)}>
                <option value="all">Tất cả ({uniqueJoints.length} loại)</option>
                {uniqueJoints.map(j => (
                  <option key={j} value={j}>{j}</option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <input
                type="text"
                placeholder="Tìm ảnh (ví dụ: train_03)..."
                value={searchImage}
                onChange={e => setSearchImage(e.target.value)}
                style={{ margin: 0, padding: '6px 10px' }}
              />
            </div>
          </div>

          <div className="review-container">
            {/* Cột trái: Danh sách cảnh báo */}
            <div className="warning-list">
              {filteredWarnings.length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>
                  Không có cảnh báo nào khớp với bộ lọc hiện tại.
                </div>
              ) : (
                filteredWarnings.map(w => (
                  <div
                    key={w.id}
                    className={`warning-card ${selectedWarning?.id === w.id ? 'active' : ''}`}
                    onClick={() => setSelectedWarning(w)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <b>{w.keypoint}</b>
                      <strong style={{ color: w.suspicion > 0.8 ? '#dc2626' : w.suspicion > 0.6 ? '#d97706' : '#2563eb' }}>
                        {Math.round(w.suspicion * 100)}%
                      </strong>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#64748b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        {renderBadge(w.warning_type)}
                        <span>{w.image_name}</span>
                      </div>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#dc2626' }}>
                        Lệch {Math.round(Math.hypot(w.human_x - w.suggested_x, w.human_y - w.suggested_y) * 10) / 10}px
                      </span>
                    </div>
                    <div style={{ fontSize: '0.76rem', color: '#64748b', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
                      <span>
                        Tin cậy: <strong style={{ color: (w.reliability ?? 1) >= 0.7 ? '#15803d' : (w.reliability ?? 1) < 0.5 ? '#b45309' : '#2563eb' }}>
                          {Math.round((w.reliability ?? 1) * 100)}%
                        </strong>
                        {w.delta !== undefined && w.delta > 0 ? (
                          <span style={{ marginLeft: 6, fontSize: '0.70rem', color: '#94a3b8' }}>
                            (δ: {Math.round(w.delta * 1000) / 1000})
                          </span>
                        ) : null}
                      </span>
                      {(w.reliability ?? 1) >= 0.7 ? (
                        <span className="badge badge-success-subtle">Tin cậy cao</span>
                      ) : (w.reliability ?? 1) < 0.5 ? (
                        <span className="badge badge-warning-subtle">Ca khó</span>
                      ) : null}
                    </div>
                    {w.review && (
                      <div style={{ marginTop: 4, fontSize: '0.75rem', color: '#16a34a' }}>
                        ✓ Đã chọn: {w.review === 'use_suggestion' ? 'Lấy theo Model' : w.review === 'keep' ? 'Giữ nguyên' : 'Bỏ qua'}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>

            {/* Cột phải: Canvas / SVG Overlay với Skeleton */}
            <div className="canvas-panel">
              {selectedWarning && imageUrl ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, fontSize: '0.88rem' }}>
                    <span>Ảnh: <b>{selectedWarning.image_name}</b> | Điểm cảnh báo: <b>{selectedWarning.keypoint}</b></span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none' }}>
                        <input
                          type="checkbox"
                          checked={showSkeleton}
                          onChange={e => setShowSkeleton(e.target.checked)}
                          style={{ margin: 0 }}
                        />
                        <span>Khung xương</span>
                      </label>
                      <div className="legend">
                        <div className="legend-item"><span className="dot dot-green"></span> Người gán</div>
                        <div className="legend-item"><span className="dot dot-red"></span> Model AI</div>
                      </div>
                    </div>
                  </div>

                  <div className="canvas-viewport">
                    <img
                      ref={imgRef}
                      src={imageUrl}
                      alt={selectedWarning.image_name}
                      onLoad={e => {
                        const target = e.currentTarget
                        setImgNaturalSize({ w: target.naturalWidth, h: target.naturalHeight })
                      }}
                    />

                    {imgNaturalSize && (
                      <svg viewBox={`0 0 ${imgNaturalSize.w} ${imgNaturalSize.h}`}>
                        {/* 1. Vẽ các đường nối khung xương (Skeleton) */}
                        {showSkeleton && imageDetails?.edges && imageDetails.edges.map(([p1Id, p2Id], idx) => {
                          const p1 = kpMap[p1Id]
                          const p2 = kpMap[p2Id]
                          if (p1 && p2 && p1.x !== null && p1.y !== null && p2.x !== null && p2.y !== null && p1.state !== 'outside' && p2.state !== 'outside') {
                            return (
                              <line
                                key={`edge-${idx}`}
                                x1={p1.x}
                                y1={p1.y}
                                x2={p2.x}
                                y2={p2.y}
                                stroke="#10b981"
                                strokeWidth="2.5"
                                strokeOpacity="0.65"
                              />
                            )
                          }
                          return null
                        })}

                        {/* 2. Vẽ tất cả các khớp người gán dạng chấm nhỏ */}
                        {showSkeleton && imageDetails?.keypoints && imageDetails.keypoints.map(kp => {
                          if (kp.x !== null && kp.y !== null && kp.state !== 'outside') {
                            return (
                              <circle
                                key={`kp-${kp.id}`}
                                cx={kp.x}
                                cy={kp.y}
                                r="3.5"
                                fill="#059669"
                                fillOpacity="0.8"
                              />
                            )
                          }
                          return null
                        })}

                        {/* 3. Đường gióng nét đứt vàng nối 2 điểm bị cảnh báo */}
                        <line
                          x1={selectedWarning.human_x}
                          y1={selectedWarning.human_y}
                          x2={selectedWarning.suggested_x}
                          y2={selectedWarning.suggested_y}
                          stroke="#f59e0b"
                          strokeWidth="3.5"
                          strokeDasharray="6,4"
                        />

                        {/* 4. Điểm người gán (Xanh lá nổi bật) */}
                        <circle
                          cx={selectedWarning.human_x}
                          cy={selectedWarning.human_y}
                          r="8"
                          fill="#22c55e"
                          stroke="#ffffff"
                          strokeWidth="2.5"
                        />

                        {/* 5. Điểm Model đề xuất (Đỏ nổi bật) */}
                        <circle
                          cx={selectedWarning.suggested_x}
                          cy={selectedWarning.suggested_y}
                          r="8"
                          fill="#ef4444"
                          stroke="#ffffff"
                          strokeWidth="2.5"
                        />
                      </svg>
                    )}
                  </div>

                  {/* Bảng phân tích chi tiết độ lệch và chuẩn OKS */}
                  <div className="canvas-info-grid">
                    <div className="info-item">
                      <span className="info-label">Độ lệch sai số</span>
                      <span className="info-value" style={{ color: '#dc2626' }}>
                        {Math.round(Math.hypot(selectedWarning.human_x - selectedWarning.suggested_x, selectedWarning.human_y - selectedWarning.suggested_y) * 10) / 10} px
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Độ tin cậy AI (R)</span>
                      <span className="info-value" style={{ color: (selectedWarning.reliability ?? 1) >= 0.7 ? '#15803d' : (selectedWarning.reliability ?? 1) < 0.5 ? '#b45309' : '#2563eb' }}>
                        {Math.round((selectedWarning.reliability ?? 1) * 100)}%
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Độ bất đồng model (δ)</span>
                      <span className="info-value">
                        {selectedWarning.delta !== undefined ? Math.round(selectedWarning.delta * 10000) / 10000 : '0.0 (K=1)'}
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Dung sai OKS (σ)</span>
                      <span className="info-value">
                        {KEYPOINT_SIGMA_MAP[selectedWarning.keypoint]?.sigma ?? 0.07} ({KEYPOINT_SIGMA_MAP[selectedWarning.keypoint]?.label ?? 'Chuẩn'})
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Người gán nhãn</span>
                      <span className="info-value">
                        ({selectedWarning.human_x}, {selectedWarning.human_y})
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Tham chiếu p* gợi ý</span>
                      <span className="info-value">
                        ({selectedWarning.suggested_x}, {selectedWarning.suggested_y})
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Phân loại lỗi</span>
                      <span className="info-value">
                        {renderBadge(selectedWarning.warning_type)}
                      </span>
                    </div>
                  </div>

                  {/* Thanh thao tác Review kèm phím tắt */}
                  <div className="action-bar">
                    <div style={{ fontSize: '0.85rem', color: '#475569' }}>
                      Độ nghi ngờ: <b>{Math.round(selectedWarning.suspicion * 100)}%</b>
                      {selectedWarning.review && (
                        <span style={{ marginLeft: 12, color: '#16a34a', fontWeight: 600 }}>
                          (Trạng thái: {selectedWarning.review})
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn-outline" onClick={() => review(selectedWarning, 'keep')}>
                        Giữ nhãn <kbd>K</kbd>
                      </button>
                      <button className="btn-success" onClick={() => review(selectedWarning, 'use_suggestion')}>
                        ✓ Lấy Model <kbd>M</kbd>
                      </button>
                      <button className="btn-outline" onClick={() => review(selectedWarning, 'skip')}>
                        Bỏ qua <kbd>S</kbd>
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', color: '#94a3b8', padding: '60px 0' }}>
                  Chọn một cảnh báo ở danh sách bên trái để xem ảnh và khung xương.
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </main>
  )
}
