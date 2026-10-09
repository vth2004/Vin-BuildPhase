import React from 'react';
import type { FrameDetail } from '../types';

interface InspectorProps {
  frame: FrameDetail | null;
  onStatusChange: (status: 'pending' | 'reviewed' | 'fixed') => void;
  onApplyModelFix?: (pointIds?: number[]) => void;
  onHoverPoints?: (points: number[]) => void;
  onLeavePoints?: () => void;
  onOpenGuidelineRule?: (ruleCode: string) => void;
}

export const Inspector: React.FC<InspectorProps> = ({
  frame,
  onStatusChange,
  onApplyModelFix,
  onHoverPoints,
  onLeavePoints,
  onOpenGuidelineRule,
}) => {
  if (!frame) {
    return (
      <aside className="panel-right" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Chưa chọn khung hình
        </div>
      </aside>
    );
  }

  const passedNme = frame.nme <= 0.035;
  const violations = frame.rule_violations || [];

  return (
    <aside className="panel-right">
      {/* 1. Header with Frame Title and Status Actions */}
      <div className="inspector-header-block">
        <div className="inspector-title-row">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span className="inspector-title">Kiểm Định #{frame.frame_index}</span>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>{frame.image_filename}</span>
          </div>
          <span
            className={`badge ${
              frame.status === 'fixed'
                ? 'badge-success'
                : frame.status === 'reviewed'
                ? 'badge-mode'
                : 'badge-warning'
            }`}
          >
            {frame.status === 'fixed'
              ? '✓ Đã sửa'
              : frame.status === 'reviewed'
              ? '👁 Đã rà soát'
              : 'Chưa sửa'}
          </span>
        </div>

        {/* Action Buttons with Keyboard Shortcuts */}
        <div className="status-actions-group">
          <button
            className={`status-btn ${frame.status === 'fixed' ? 'active-fixed' : ''}`}
            onClick={() => onStatusChange('fixed')}
            title="Đánh dấu đã sửa / giữ nguyên nhãn (Phím K)"
          >
            ✓ Giữ / Đã sửa <kbd>K</kbd>
          </button>
          <button
            className="status-btn active-reviewed"
            onClick={() => onApplyModelFix && onApplyModelFix()}
            title="Tự động sửa lỗi theo tọa độ Model AI (Phím M)"
          >
            ⚡ Lấy Model AI <kbd>M</kbd>
          </button>
          <button
            className={`status-btn ${frame.status === 'pending' ? 'active-pending' : ''}`}
            onClick={() => onStatusChange('pending')}
            title="Đánh dấu chưa sửa / bỏ qua (Phím S)"
          >
            ⚪ Bỏ qua <kbd>S</kbd>
          </button>
        </div>
      </div>

      {/* 2. Key Metrics Row */}
      <div className="metrics-row">
        <div className="metric-tile">
          <span className="metric-tile-label">NME Toàn Ảnh</span>
          <span
            className="metric-tile-val"
            style={{ color: passedNme ? 'var(--success)' : 'var(--danger)' }}
          >
            {frame.nme.toFixed(4)}
          </span>
          <span
            className="metric-tile-sub"
            style={{ color: passedNme ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}
          >
            {passedNme ? 'ĐẠT (≤ 0.035)' : 'VƯỢT NGƯỠNG'}
          </span>
        </div>

        <div className="metric-tile">
          <span className="metric-tile-label">Khoảng Cách Mắt (IOD)</span>
          <span className="metric-tile-val">{Math.round(frame.iod)} px</span>
          <span className="metric-tile-sub">
            {frame.iod >= 80 ? 'Cự ly: GẦN' : frame.iod >= 38 ? 'Cự ly: VỪA' : 'Cự ly: XA'}
          </span>
        </div>

        <div className="metric-tile">
          <span className="metric-tile-label">Vi Phạm</span>
          <span
            className="metric-tile-val"
            style={{ color: violations.length > 0 ? 'var(--danger)' : 'var(--success)' }}
          >
            {violations.length}
          </span>
          <span className="metric-tile-sub">
            {violations.length > 0 ? 'Quy chuẩn' : 'Hoàn hảo'}
          </span>
        </div>
      </div>

      {/* 2.1 Scale-Invariant Adaptive Tolerance Card & Point State Breakdown */}
      {(() => {
        const kptList = Object.values(frame.human_keypoints || {});
        const visCount = kptList.filter((k) => k.state === 'visible' || !k.state).length;
        const occCount = kptList.filter((k) => k.state === 'occluded').length;
        const outCount = kptList.filter((k) => k.state === 'outside').length;
        const anchorTol = (frame.iod * 0.03).toFixed(1);
        const contourTol = (frame.iod * 0.05).toFixed(1);
        const isFar = frame.iod < 38;

        return (
          <div style={{ padding: '0 14px 10px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {/* Multi-person Cabin Face Lock Badge */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '7px 10px',
                background: '#f1f5f9',
                borderRadius: '6px',
                border: '1px solid #e2e8f0',
                fontSize: '0.74rem',
              }}
            >
              <span style={{ fontWeight: 600, color: '#334155' }}>
                🎯 Khóa Mục Tiêu: <span style={{ color: '#0284c7' }}>Mặt Tài Xế (CVAT Target)</span>
              </span>
              <span style={{ fontSize: '0.68rem', color: '#64748b' }}>
                {kptList.length} điểm khớp
              </span>
            </div>

            {/* Scale-Invariant Dynamic Tolerance Box */}
            <div
              style={{
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '9px 11px',
                fontSize: '0.73rem',
                color: '#334155',
                lineHeight: '1.45',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontWeight: 700, color: '#0f172a' }}>
                  📏 Chuẩn Hóa Dung Sai Theo Cự Ly ({isFar ? 'Mặt Ở Xa' : 'Mặt Ở Gần'})
                </span>
                <span
                  style={{
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '0.66rem',
                    fontWeight: 700,
                    background: isFar ? '#fef3c7' : '#dcfce7',
                    color: isFar ? '#92400e' : '#166534',
                  }}
                >
                  IOD: {Math.round(frame.iod)}px
                </span>
              </div>

              <div style={{ display: 'flex', gap: '12px', color: '#475569', margin: '4px 0' }}>
                <span>• Điểm neo (≤3% IOD): <strong>≤ {anchorTol}px</strong></span>
                <span>• Điểm viền (≤5% IOD): <strong>≤ {contourTol}px</strong></span>
              </div>

              <div style={{ fontSize: '0.70rem', color: '#64748b', fontStyle: 'italic', borderTop: '1px dashed #e2e8f0', paddingTop: '4px', marginTop: '4px' }}>
                💡 {isFar
                  ? 'Khuôn mặt nhỏ/ở xa: Lệch 10px tương đương ~45% IOD (lỗi nghiêm trọng, vượt ngưỡng cho phép 5%).'
                  : 'Khuôn mặt to/ở gần: Lệch 10px chỉ tương đương ~8% IOD (nằm gần ngưỡng dung sai).'}
              </div>
            </div>

            {/* Point States Breakdown */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px' }}>
              <div
                style={{
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '6px',
                  padding: '5px 8px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.67rem', color: '#64748b' }}>👁 Nhìn Rõ</div>
                <div style={{ fontWeight: 700, color: '#10b981', fontSize: '0.85rem' }}>{visCount}</div>
              </div>
              <div
                style={{
                  background: occCount > 0 ? '#fffbeb' : '#f8fafc',
                  border: `1px solid ${occCount > 0 ? '#fde68a' : '#e2e8f0'}`,
                  borderRadius: '6px',
                  padding: '5px 8px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.67rem', color: occCount > 0 ? '#b45309' : '#64748b' }}>🕶 Bị Che (Kính)</div>
                <div style={{ fontWeight: 700, color: occCount > 0 ? '#d97706' : '#64748b', fontSize: '0.85rem' }}>{occCount}</div>
              </div>
              <div
                style={{
                  background: outCount > 0 ? '#fef2f2' : '#f8fafc',
                  border: `1px solid ${outCount > 0 ? '#fecaca' : '#e2e8f0'}`,
                  borderRadius: '6px',
                  padding: '5px 8px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.67rem', color: outCount > 0 ? '#b91c1c' : '#64748b' }}>🚫 Ngoài Khung</div>
                <div style={{ fontWeight: 700, color: outCount > 0 ? '#ef4444' : '#64748b', fontSize: '0.85rem' }}>{outCount}</div>
              </div>
            </div>

            {/* Hard Case / Occlusion Banner */}
            {frame.case_type && frame.case_type !== 'normal' && (
              <div
                style={{
                  background: '#fffbeb',
                  border: '1px solid #fde68a',
                  borderRadius: '6px',
                  padding: '7px 10px',
                  fontSize: '0.73rem',
                  lineHeight: '1.4',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                  <span style={{ fontWeight: 700, color: '#b45309' }}>
                    {frame.case_label || '🕶️ Ca khó / Góc khuất'}
                  </span>
                  <span
                    style={{
                      background: '#fef3c7',
                      color: '#92400e',
                      padding: '1px 6px',
                      borderRadius: '4px',
                      fontSize: '0.67rem',
                      fontWeight: 700,
                    }}
                  >
                    Tin cậy AI: {Math.round((frame.ai_reliability ?? 0.5) * 100)}%
                  </span>
                </div>
                <div style={{ color: '#78350f', fontSize: '0.70rem' }}>
                  {frame.case_description || 'Khuôn mặt có góc nghiêng hoặc bị che khuất. Dung sai giải phẫu được nới lỏng.'}
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* 3. Violations & CVAT Step-by-Step Fix List */}
      <div className="inspector-body">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="section-label">Danh Sách Lỗi Cần Sửa ({violations.length})</span>
          {violations.length > 0 && (
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Rê chuột vào điểm để xem trên ảnh
            </span>
          )}
        </div>

        {/* Quick-Fix All Button (Model AI) */}
        {violations.length > 0 && frame.has_model_prediction === 1 && (
          <button
            type="button"
            className="btn-quickfix-all"
            onClick={() => onApplyModelFix && onApplyModelFix()}
            title="Tự động sửa tất cả các điểm có vi phạm trong khung hình này theo Model AI (Phím M)"
          >
            ⚡ Sửa nhanh tất cả lỗi theo Model AI <kbd>M</kbd>
          </button>
        )}

        {violations.length === 0 ? (
          <div
            style={{
              padding: '24px 16px',
              textAlign: 'center',
              background: 'var(--success-bg)',
              border: '1px solid var(--success-border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--success)',
              fontSize: '0.82rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
            }}
          >
            <div style={{ fontSize: '24px' }}>✨</div>
            <div style={{ fontWeight: 700 }}>Khung hình đạt chuẩn tuyệt đối!</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Không phát hiện vi phạm nào theo 14 quy tắc Guideline VF-50 v1.3.
            </div>
          </div>
        ) : (
          violations.map((v, idx) => {
            const isCritical = v.level === 'ERROR' || v.severity === 'critical';

            return (
              <div
                key={`violation_card_${idx}`}
                className={`violation-card-modern ${isCritical ? 'critical' : ''}`}
                onMouseEnter={() => onHoverPoints && onHoverPoints(v.points || [])}
                onMouseLeave={() => onLeavePoints && onLeavePoints()}
              >
                {/* Head */}
                <div className="violation-head">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <button
                      type="button"
                      className="violation-code-badge clickable"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpenGuidelineRule && onOpenGuidelineRule(v.code);
                      }}
                      title={`Xem chi tiết quy chuẩn [${v.code}] trong Sổ tay Guideline`}
                    >
                      {v.code} ↗
                    </button>
                    <span
                      className={`badge ${isCritical ? 'badge-error' : 'badge-warning'}`}
                      style={{ fontSize: '0.66rem' }}
                    >
                      {isCritical ? 'CRITICAL' : 'WARNING'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {v.gl_section || 'Guideline VF-50'}
                    </span>
                    {onOpenGuidelineRule && (
                      <button
                        type="button"
                        className="btn-lookup-rule"
                        onClick={(e) => {
                          e.stopPropagation();
                          onOpenGuidelineRule(v.code);
                        }}
                        title={`Xem chi tiết luật ${v.code}`}
                      >
                        📖 Tra cứu
                      </button>
                    )}
                  </div>
                </div>

                {/* Rule Title & Description */}
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.82rem' }}>
                  {v.rule_name || v.message}
                </div>
                {v.rule_name && v.message && v.rule_name !== v.message && (
                  <div className="violation-desc">{v.message}</div>
                )}

                {/* Points affected */}
                {v.points && v.points.length > 0 && (
                  <div className="violation-points-row">
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Điểm:</span>
                    {v.points.slice(0, 10).map((p) => (
                      <span
                        key={`pt_chip_${p}`}
                        className="point-chip"
                        onMouseEnter={() => onHoverPoints && onHoverPoints([p])}
                        onMouseLeave={() => onLeavePoints && onLeavePoints()}
                      >
                        #{p}
                      </span>
                    ))}
                    {v.points.length > 10 && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        +{v.points.length - 10} điểm khác
                      </span>
                    )}
                  </div>
                )}

                {/* Direct Instruction for CVAT Hand-Labeling */}
                <div className="cvat-fix-guide-box">
                  <span className="cvat-fix-guide-title">
                    👉 Hướng dẫn sửa trên CVAT:
                  </span>
                  <span>
                    {v.code === 'R01'
                      ? 'Thiếu điểm keypoint trong bộ 50 điểm. Kiểm tra lại nhãn bị sót hoặc bị xoá nhầm.'
                      : v.code === 'R02'
                      ? 'Cả 50 điểm đều là Visible. Kiểm tra xem có điểm nào bị tóc, gọng kính hay góc quay che khuất không để gắn nhãn Occluded/Outside.'
                      : v.code === 'R03'
                      ? 'Quy ước VinFast: Bên TRÁI ảnh (x nhỏ hơn) luôn là longmaytrai / mattrai; bên PHẢI ảnh (x lớn hơn) là longmayphai / matphai.'
                      : v.code === 'R04'
                      ? 'Tọa độ x của lông mày phải tăng dần từ 0 đến 9. Kiểm tra xem có nối nhầm thứ tự điểm không.'
                      : v.code === 'R05'
                      ? 'Khóe ngoài mắt trái (14) phải có x nhỏ nhất, khóe trong (18) có x lớn nhất. Tương tự mắt phải 22 và 26.'
                      : v.code === 'R06'
                      ? 'Mí trên (15, 16, 17 hoặc 23, 24, 25) bị kéo thấp hơn mí dưới (21, 20, 19 hoặc 29, 28, 27). Khi tài xế nhắm mắt, mí trên và mí dưới nằm sát nhau, tuyệt đối không để mí trên lật xuống dưới.'
                      : v.code === 'R07'
                      ? 'Đa giác mắt bị tự cắt thành hình chữ X. Kiểm tra thứ tự các điểm quanh mí mắt.'
                      : v.code === 'R08'
                      ? 'Ba đoạn sống mũi (10-11, 11-12, 12-13) phải chia đều nhau. Điểm 13 là gốc chân sống mũi (ngang khóe mũi), không được kéo xuống chóp mũi.'
                      : v.code === 'R09'
                      ? 'Bờ môi trong (42-49) phải nằm hoàn toàn trong bờ môi ngoài (30-41). Không để điểm môi trong trồi ra ngoài.'
                      : v.code === 'R10'
                      ? 'Mục 4.2 GL: Nếu một bộ phận (mắt < 4/8 điểm, lông mày < 3/5 điểm) bị che khuất thì bắt buộc phải đánh Outside cho TOÀN BỘ bộ phận đó.'
                      : v.code === 'R11'
                      ? 'Điểm rơi ra ngoài biên ảnh bắt buộc phải đổi thuộc tính state thành "outside".'
                      : v.code === 'R12'
                      ? 'Hai điểm khác nhau không được trùng khít tọa độ nhau khi cả hai đều Visible.'
                      : v.code === 'R13'
                      ? 'Điểm nhảy bất thường so với frame trước. Kiểm tra độ ổn định nhãn giữa các frame liên tiếp.'
                      : v.code === 'R14'
                      ? 'Điểm Occluded (kính râm, tóc che) bị đặt sai vị trí giải phẫu (> 15% IOD). Cần ước lượng lại vị trí hợp lý dựa trên cấu trúc xương mặt.'
                      : 'Mở job CVAT, tìm đến khung hình này và căn chỉnh lại tọa độ điểm theo đúng quy chuẩn.'}
                  </span>
                </div>

                {/* Per-violation quick-fix button */}
                {frame.has_model_prediction === 1 && v.points && v.points.length > 0 && (
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '7px' }}>
                    <button
                      type="button"
                      className="btn-quickfix-rule"
                      onClick={(e) => {
                        e.stopPropagation();
                        onApplyModelFix && onApplyModelFix(v.points);
                      }}
                      title={`Tự động sửa ${v.points.length} điểm này theo tọa độ Model AI`}
                    >
                      ⚡ Sửa {v.points.length} điểm này theo AI
                    </button>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
