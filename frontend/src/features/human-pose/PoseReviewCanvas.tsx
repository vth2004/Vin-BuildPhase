import { useMemo, useRef, useState } from 'react'
import type { ImageDetails, KeypointDetail, Warning } from './types'
import { KEYPOINT_SIGMA_MAP } from './constants'
import { renderBadge } from './WarningBadge'

type Props = {
  selectedWarning: Warning | null
  imageUrl: string | null
  imageDetails: ImageDetails | null
  review: (warning: Warning, value: string) => Promise<void>
}

export default function PoseReviewCanvas({ selectedWarning, imageUrl, imageDetails, review }: Props) {
  const [imgNaturalSize, setImgNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const imgRef = useRef<HTMLImageElement>(null)
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

  return (
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
  )
}
