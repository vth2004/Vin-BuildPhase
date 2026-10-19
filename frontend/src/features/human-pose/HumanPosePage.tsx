import { clearSession } from '../../shared/session'
import { useHumanPoseWorkspace } from './useHumanPoseWorkspace'
import PoseReviewCanvas from './PoseReviewCanvas'
import { renderBadge } from './WarningBadge'

export default function HumanPosePage() {
  const { session, setSession, datasets, file, validationError, setActiveDatasetId, warnings, selectedWarning, setSelectedWarning, message, imageDetails, runMode, aiDraftRunId, minSuspicion, setMinSuspicion, statusFilter, setStatusFilter, jointFilter, setJointFilter, searchImage, setSearchImage, reliabilityFilter, setReliabilityFilter, uniqueJoints, filteredWarnings, createProject, resumeProject, handleZipSelect, handleAnnotationSelect, upload, startRun, downloadAiDraft, review, exportCleaned, imageUrl, } = useHumanPoseWorkspace()

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
            <PoseReviewCanvas selectedWarning={selectedWarning} imageUrl={imageUrl} imageDetails={imageDetails} review={review} />
          </div>
        </section>
      )}
    </main>
  )
}
