import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

type Dataset = { id: string; name: string; status: string; image_count: number; schema_id: string }

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

type Session = { projectId: string; recoveryKey: string; projectName: string }

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

const saved = (): Session | null => {
  const raw = localStorage.getItem('landmark-session')
  return raw ? JSON.parse(raw) : null
}

function App() {
  const [session, setSession] = useState<Session | null>(saved())
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [schemaId, setSchemaId] = useState('vf_humanpose17_v1')
  const [runId, setRunId] = useState<string | null>(null)
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<Warning[]>([])
  const [selectedWarning, setSelectedWarning] = useState<Warning | null>(null)
  const [message, setMessage] = useState('')
  const [imgNaturalSize, setImgNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [imageDetails, setImageDetails] = useState<ImageDetails | null>(null)
  const [showSkeleton, setShowSkeleton] = useState(true)

  // Bộ lọc
  const [minSuspicion, setMinSuspicion] = useState<number>(0.2)
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'reviewed'>('all')
  const [jointFilter, setJointFilter] = useState<string>('all')
  const [searchImage, setSearchImage] = useState<string>('')

  const imgRef = useRef<HTMLImageElement>(null)

  const headers = () => ({ Authorization: `Bearer ${session?.recoveryKey}` })

  const request = async (path: string, init: RequestInit = {}) => {
    const res = await fetch(`${API}${path}`, { ...init, headers: { ...headers(), ...(init.headers || {}) } })
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail || 'Request failed')
    return res.json()
  }

  const loadProject = async () => {
    if (!session) return
    const data = await request(`/projects/${session.projectId}`)
    setDatasets(data.datasets)
    if (data.datasets.length > 0 && !activeDatasetId) {
      setActiveDatasetId(data.datasets[0].id)
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
      setMessage(`Tiến độ Run: ${run.status} (${run.progress}%)`)
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
      return true
    })
  }, [warnings, minSuspicion, statusFilter, jointFilter, searchImage])

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
      localStorage.setItem('landmark-session', JSON.stringify(next))
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
      localStorage.setItem('landmark-session', JSON.stringify(next))
      setSession(next)
      setMessage(`Đã mở lại project: ${data.project.name}`)
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const upload = async () => {
    if (!file || !session) return
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
      setActiveDatasetId(data.id)
      setMessage(`Đã nhận dataset ${data.name}: ${data.image_count} ảnh theo ${data.schema_id}.`)
      loadProject()
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  const startRun = async (datasetId: string) => {
    try {
      setActiveDatasetId(datasetId)
      const data = await request(`/projects/${session!.projectId}/runs?dataset_id=${datasetId}`, { method: 'POST' })
      setRunId(data.id)
      setWarnings([])
      setSelectedWarning(null)
      setMessage(`Đã bắt đầu chạy Run (${data.mode === 'real' ? 'Model YOLO26s-pose thật' : 'Mock'})...`)
    } catch (e) {
      setMessage((e as Error).message)
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

  if (!session) {
    return (
      <main>
        <h1>Landmark QA Web</h1>
        <p>Hệ thống tự động phát hiện điểm nghi ngờ gán nhãn sai trên dữ liệu Landmark/Pose.</p>
        <section>
          <h2>Tạo dự án mới</h2>
          <form onSubmit={createProject}>
            <input name="name" required placeholder="Tên dự án mới" />
            <button>Tạo project</button>
          </form>
        </section>

        <section>
          <h2>Hoặc mở lại dự án đã có</h2>
          <form onSubmit={resumeProject}>
            <input name="projectId" required placeholder="Project ID" />
            <input name="recoveryKey" required placeholder="Recovery Key" />
            <button className="btn-outline">Mở lại project</button>
          </form>
        </section>
      </main>
    )
  }

  const imageUrl = selectedWarning && activeDatasetId
    ? `${API}/projects/${session.projectId}/datasets/${activeDatasetId}/images/${selectedWarning.image_name}?token=${session.recoveryKey}`
    : null

  // Tạo map id -> keypoint để vẽ skeleton
  const kpMap = useMemo(() => {
    const map: Record<number, KeypointDetail> = {}
    if (imageDetails?.keypoints) {
      imageDetails.keypoints.forEach(k => {
        map[k.id] = k
      })
    }
    return map
  }, [imageDetails])

  return (
    <main>
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
            localStorage.removeItem('landmark-session')
            setSession(null)
          }}
        >
          Rời project
        </button>
      </header>

      {message && <p className="notice">{message}</p>}

      {/* 1. Upload Dataset */}
      <section>
        <h2>1. Upload Dataset đã gán nhãn</h2>
        <p style={{ fontSize: '0.88rem', color: '#64748b', marginTop: 0 }}>
          Hỗ trợ file ZIP xuất trực tiếp từ CVAT (gồm cả ảnh & nhãn), hoặc ZIP ảnh kèm file JSON nhãn riêng.
        </p>
        <select value={schemaId} onChange={e => setSchemaId(e.target.value)}>
          <option value="vf_humanpose17_v1">VinFast HumanPose-17 (COCO 17 điểm)</option>
          <option value="vf_face_landmark50_v1">VinFast Face Landmark VF-50</option>
        </select>
        <input type="file" accept=".zip,application/zip" onChange={e => setFile(e.target.files?.[0] || null)} />
        <input type="file" accept=".json,application/json" onChange={e => setAnnotationFile(e.target.files?.[0] || null)} />
        <button disabled={!file} onClick={upload}>
          Tải lên dataset
        </button>
      </section>

      {/* 2. Quản lý Dataset & Khởi chạy Run */}
      <section>
        <h2>2. Danh sách Dataset & Quét lỗi với Model AI</h2>
        {datasets.length === 0 ? (
          <p style={{ color: '#64748b' }}>Chưa có dataset nào được tải lên.</p>
        ) : (
          <div className="dataset-list">
            {datasets.map(d => (
              <div className="dataset-item" key={d.id}>
                <span>
                  <strong>{d.name}</strong> · {d.image_count} ảnh · Schema: <code>{d.schema_id}</code>
                </span>
                <button onClick={() => startRun(d.id)}>
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
                      <span className="info-label">Model AI gợi ý</span>
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

createRoot(document.getElementById('root')!).render(<App />)
