import React, { useState, useMemo } from 'react';
import type { FrameDetail } from '../types';
import {
  POINT_ANATOMY_NAMES,
  ANCHOR_POINTS,
  getPointColor,
  getSkeletonByPointId,
} from '../vf50_constants';

interface InspectorProps {
  frame: FrameDetail | null;
  onStatusChange: (status: 'pending' | 'reviewed' | 'fixed') => void;
  onApplyModelFix?: (pointIds?: number[]) => void;
  onHoverPoints?: (points: number[]) => void;
  onLeavePoints?: () => void;
  onOpenGuidelineRule?: (ruleCode: string) => void;
  onSetPointSource?: (pointId: number, source: 'human' | 'model') => void;
  selectedPointId?: number | null;
  onSelectPoint?: (pointId: number | null) => void;
}

export const Inspector: React.FC<InspectorProps> = ({
  frame,
  onStatusChange,
  onApplyModelFix,
  onHoverPoints,
  onLeavePoints,
  onOpenGuidelineRule,
  onSetPointSource,
  selectedPointId,
  onSelectPoint,
}) => {
  const [activeTab, setActiveTab] = useState<'violations' | 'points'>('violations');
  const [pointFilter, setPointFilter] = useState<'all' | 'issues' | 'model' | 'eyes' | 'lips' | 'nose' | 'brows'>('all');

  const passedNme = (frame?.nme ?? 0) <= 0.035;
  const violations = frame?.rule_violations || [];
  const humanKpts = frame?.human_keypoints || {};
  const modelKpts = frame?.model_keypoints || {};
  const pointSources = frame?.point_sources || {};

  // Pre-calculate per-point stats for 50 keypoints
  const pointsMeta = useMemo(() => {
    if (!frame) return [];
    const list = [];
    const violatedPointIds = new Set<number>();
    violations.forEach((v) => {
      v.points?.forEach((p) => violatedPointIds.add(p));
    });

    for (let id = 0; id < 50; id++) {
      const h = humanKpts[id] || humanKpts[String(id)];
      const m = modelKpts[id] || modelKpts[String(id)];
      const isAnchor = ANCHOR_POINTS.has(id);
      const maxTolPct = isAnchor ? 3.0 : 5.0;
      let distPx: number | null = null;
      let pctIod: number | null = null;
      let isExceeded = false;

      if (h && m && h.x !== null && h.y !== null && m.x !== null && m.y !== null) {
        distPx = Math.hypot(h.x - m.x, h.y - m.y);
        if (frame.iod && frame.iod > 0) {
          pctIod = (distPx / frame.iod) * 100;
          isExceeded = pctIod > maxTolPct;
        }
      }

      const isViolated = violatedPointIds.has(id);
      const currentSource: 'human' | 'model' = pointSources[String(id)] || pointSources[id] || 'human';
      const sk = getSkeletonByPointId(id);

      list.push({
        id,
        name: POINT_ANATOMY_NAMES[id] || `Điểm #${id}`,
        isAnchor,
        maxTolPct,
        distPx,
        pctIod,
        isExceeded,
        isViolated,
        currentSource,
        h,
        m,
        skeleton: sk?.name || '',
        skeletonLabel: sk?.label || '',
      });
    }
    return list;
  }, [frame, humanKpts, modelKpts, pointSources, violations]);

  // Point counts
  const modelCount = pointsMeta.filter((p) => p.currentSource === 'model').length;
  const humanCount = 50 - modelCount;
  const issuesCount = pointsMeta.filter((p) => p.isViolated || p.isExceeded).length;

  // Filtered points for the Point Triage tab
  const filteredPoints = useMemo(() => {
    return pointsMeta.filter((p) => {
      if (pointFilter === 'issues') return p.isViolated || p.isExceeded;
      if (pointFilter === 'model') return p.currentSource === 'model';
      if (pointFilter === 'eyes') return (p.id >= 14 && p.id <= 29);
      if (pointFilter === 'lips') return (p.id >= 30 && p.id <= 49);
      if (pointFilter === 'nose') return (p.id >= 10 && p.id <= 13);
      if (pointFilter === 'brows') return (p.id >= 0 && p.id <= 9);
      return true;
    });
  }, [pointsMeta, pointFilter]);

  if (!frame) {
    return (
      <aside className="panel-right" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Chưa chọn khung hình
        </div>
      </aside>
    );
  }

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
            ✓ Đã sửa <kbd>K</kbd>
          </button>
          <button
            className="status-btn active-reviewed"
            onClick={() => onApplyModelFix && onApplyModelFix()}
            title="Tự động sửa toàn bộ lỗi theo tọa độ Model AI (Phím M)"
          >
            ⚡ Lấy Hết AI <kbd>M</kbd>
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
          <span className="metric-tile-label">Nguồn 50 Điểm</span>
          <div style={{ display: 'flex', gap: '6px', alignItems: 'baseline', marginTop: '2px' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#16a34a' }}>👤 {humanCount}</span>
            <span style={{ color: '#94a3b8' }}>/</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0284c7' }}>🤖 {modelCount}</span>
          </div>
          <span className="metric-tile-sub">
            {modelCount > 0 ? `${modelCount} điểm lấy AI` : '100% nhãn người'}
          </span>
        </div>
      </div>

      {/* 2.1 Tab Selector: Vi Phạm vs Lựa Chọn Từng Điểm */}
      <div className="inspector-tabs-container">
        <button
          type="button"
          className={`inspector-tab-btn ${activeTab === 'violations' ? 'active' : ''}`}
          onClick={() => setActiveTab('violations')}
        >
          <span>⚠️ Vi Phạm Quy Chuẩn</span>
          <span className={`tab-count-badge ${violations.length > 0 ? 'badge-error-count' : ''}`}>
            {violations.length}
          </span>
        </button>

        <button
          type="button"
          className={`inspector-tab-btn ${activeTab === 'points' ? 'active' : ''}`}
          onClick={() => setActiveTab('points')}
        >
          <span>🎯 Lựa Chọn Từng Điểm</span>
          <span className="tab-count-badge">
            {issuesCount > 0 ? `${issuesCount} lệch` : '50'}
          </span>
        </button>
      </div>

      {/* 3. TAB CONTENT */}
      {activeTab === 'violations' ? (
        /* TAB 1: VIOLATIONS VIEW WITH PER-POINT CONTROLS */
        <div className="inspector-body">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="section-label">Danh Sách Lỗi ({violations.length})</span>
            {violations.length > 0 && (
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Rê chuột để soi điểm trên ảnh
              </span>
            )}
          </div>

          {/* Quick Fix All Button */}
          {violations.length > 0 && frame.has_model_prediction === 1 && (
            <button
              type="button"
              className="btn-quickfix-all"
              onClick={() => onApplyModelFix && onApplyModelFix()}
              title="Sửa tất cả các điểm có vi phạm sang Model AI (Phím M)"
            >
              ⚡ Sửa tất cả lỗi theo Model AI <kbd>M</kbd>
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
                Không phát hiện vi phạm nào theo 14 quy tắc Guideline VF-50.
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

                  {/* Rule Title & Message */}
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.82rem' }}>
                    {v.rule_name || v.message}
                  </div>
                  {v.rule_name && v.message && v.rule_name !== v.message && (
                    <div className="violation-desc">{v.message}</div>
                  )}

                  {/* CVAT Fix Guide Text */}
                  <div className="cvat-fix-guide-box">
                    <span className="cvat-fix-guide-title">👉 Hướng dẫn chuẩn hóa:</span>
                    <span>
                      {v.code === 'R01'
                        ? 'Thiếu điểm keypoint. Kiểm tra nhãn bị sót hoặc bị xoá nhầm.'
                        : v.code === 'R02'
                        ? 'Cả 50 điểm đều Visible. Gán Occluded/Outside cho điểm bị che khuất.'
                        : v.code === 'R03'
                        ? 'Bên TRÁI ảnh (x nhỏ hơn) luôn là bên Trái; bên PHẢI ảnh (x lớn hơn) là bên Phải.'
                        : v.code === 'R04'
                        ? 'Tọa độ x của lông mày phải tăng dần từ đuôi đến đầu.'
                        : v.code === 'R05'
                        ? 'Khóe ngoài mắt phải có x ngoài cùng, khóe trong có x trong cùng.'
                        : v.code === 'R06'
                        ? 'Mí trên bị kéo thấp hơn mí dưới (Lật mí). Căn chỉnh mí trên nằm trên mí dưới.'
                        : v.code === 'R07'
                        ? 'Đa giác mắt bị tự cắt thành hình chữ X. Kiểm tra thứ tự nối điểm mí mắt.'
                        : v.code === 'R08'
                        ? 'Ba đoạn sống mũi phải chia đều. Điểm 13 ở chân sống mũi ngang cánh mũi.'
                        : v.code === 'R09'
                        ? 'Bờ môi trong phải nằm hoàn toàn trong bờ môi ngoài.'
                        : v.code === 'R10'
                        ? 'Mục 4.2 GL: Bộ phận che quá ngưỡng phải đánh Outside toàn bộ bộ phận.'
                        : v.code === 'R11'
                        ? 'Điểm ngoài biên ảnh phải đổi state sang Outside.'
                        : v.code === 'R12'
                        ? 'Hai điểm khác nhau không được trùng khít tọa độ.'
                        : v.code === 'R13'
                        ? 'Điểm nhảy bất thường so với frame trước.'
                        : v.code === 'R14'
                        ? 'Điểm Occluded bị đặt sai giải phẫu (>15% IOD). Ước lượng lại cấu trúc xương mặt.'
                        : 'Kiểm tra và căn chỉnh lại tọa độ điểm theo đúng quy chuẩn giải phẫu.'}
                    </span>
                  </div>

                  {/* Granular Point Selection for THIS Violation */}
                  {v.points && v.points.length > 0 && (
                    <div className="violation-point-actions-panel">
                      <div className="violation-point-actions-header">
                        <span>🎯 Chọn nguồn từng điểm trong lỗi này:</span>
                        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                          Click điểm để chọn nguồn Người / AI
                        </span>
                      </div>

                      <div className="violation-point-list">
                        {v.points.map((pId) => {
                          const pMeta = pointsMeta.find((p) => p.id === pId);
                          const currentSrc = pMeta?.currentSource || 'human';
                          const isAnchor = ANCHOR_POINTS.has(pId);
                          const isSelected = selectedPointId === pId;

                          return (
                            <div
                              key={`v_pt_${idx}_${pId}`}
                              className={`point-action-row ${isSelected ? 'selected' : ''}`}
                              onClick={() => onSelectPoint && onSelectPoint(pId)}
                              onMouseEnter={() => onHoverPoints && onHoverPoints([pId])}
                              onMouseLeave={() => onLeavePoints && onLeavePoints()}
                            >
                              <div className="point-action-left">
                                <span
                                  className="point-badge"
                                  style={{ background: getPointColor(pId), color: '#ffffff' }}
                                >
                                  #{pId}
                                </span>
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                  <span className="point-name-text">
                                    {POINT_ANATOMY_NAMES[pId] || `Điểm #${pId}`}
                                  </span>
                                  <div style={{ display: 'flex', gap: '6px', fontSize: '0.67rem', color: '#64748b' }}>
                                    <span>{isAnchor ? '⭐ Neo (≤3%)' : 'Viền (≤5%)'}</span>
                                    {pMeta?.distPx !== null && (
                                      <span
                                        style={{
                                          color: pMeta?.isExceeded ? '#dc2626' : '#16a34a',
                                          fontWeight: 600,
                                        }}
                                      >
                                        Δ {pMeta?.distPx?.toFixed(1)}px ({pMeta?.pctIod?.toFixed(1)}% IOD)
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>

                              {/* Toggle Buttons: Human vs Model */}
                              <div className="point-choice-buttons">
                                <button
                                  type="button"
                                  className={`btn-choice btn-choice-human ${currentSrc === 'human' ? 'active' : ''}`}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onSetPointSource && onSetPointSource(pId, 'human');
                                  }}
                                  title="Giữ tọa độ do Người dán nhãn ban đầu"
                                >
                                  👤 Người
                                </button>
                                <button
                                  type="button"
                                  className={`btn-choice btn-choice-model ${currentSrc === 'model' ? 'active' : ''}`}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onSetPointSource && onSetPointSource(pId, 'model');
                                  }}
                                  title="Áp dụng tọa độ gợi ý từ Model AI"
                                >
                                  🤖 AI
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      ) : (
        /* TAB 2: DEDICATED 50 POINTS TRIAGE TABLE */
        <div className="inspector-body">
          {/* Filter Pills */}
          <div className="point-triage-filters">
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'all' ? 'active' : ''}`}
              onClick={() => setPointFilter('all')}
            >
              Tất cả (50)
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'issues' ? 'active' : ''}`}
              onClick={() => setPointFilter('issues')}
            >
              ⚠️ Lỗi/Lệch ({issuesCount})
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'model' ? 'active' : ''}`}
              onClick={() => setPointFilter('model')}
            >
              🤖 Đã lấy AI ({modelCount})
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'eyes' ? 'active' : ''}`}
              onClick={() => setPointFilter('eyes')}
            >
              Mắt (16)
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'lips' ? 'active' : ''}`}
              onClick={() => setPointFilter('lips')}
            >
              Môi (20)
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'nose' ? 'active' : ''}`}
              onClick={() => setPointFilter('nose')}
            >
              Mũi (4)
            </button>
            <button
              type="button"
              className={`filter-pill ${pointFilter === 'brows' ? 'active' : ''}`}
              onClick={() => setPointFilter('brows')}
            >
              Lông mày (10)
            </button>
          </div>

          {/* Quick Batch Actions for Points */}
          <div className="point-triage-batch-actions">
            <button
              type="button"
              className="btn-batch-sub"
              onClick={() => {
                const issueIds = pointsMeta
                  .filter((p) => p.isViolated || p.isExceeded)
                  .map((p) => p.id);
                if (issueIds.length > 0 && onApplyModelFix) {
                  onApplyModelFix(issueIds);
                }
              }}
              disabled={issuesCount === 0}
              title="Đổi tất cả các điểm vi phạm/lệch sang Model AI"
            >
              🤖 Lấy AI cho các điểm lệch ({issuesCount})
            </button>
            <button
              type="button"
              className="btn-batch-sub outline"
              onClick={() => {
                // Revert all points to human
                pointsMeta.forEach((p) => {
                  if (p.currentSource === 'model' && onSetPointSource) {
                    onSetPointSource(p.id, 'human');
                  }
                });
              }}
              disabled={modelCount === 0}
              title="Khôi phục toàn bộ các điểm về nhãn Người ban đầu"
            >
              👤 Khôi phục tất cả về Người
            </button>
          </div>

          {/* Point Cards List */}
          <div className="point-triage-list">
            {filteredPoints.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                Không có điểm nào theo bộ lọc đã chọn
              </div>
            ) : (
              filteredPoints.map((pt) => {
                const isSelected = selectedPointId === pt.id;
                const hCoord = pt.h ? `(${pt.h.x?.toFixed(1) ?? '--'}, ${pt.h.y?.toFixed(1) ?? '--'})` : 'Chưa có';
                const mCoord = pt.m ? `(${pt.m.x?.toFixed(1) ?? '--'}, ${pt.m.y?.toFixed(1) ?? '--'})` : 'Chưa có';

                return (
                  <div
                    key={`point_card_${pt.id}`}
                    className={`point-triage-card ${isSelected ? 'selected' : ''} ${pt.isExceeded ? 'has-issue' : ''}`}
                    onClick={() => onSelectPoint && onSelectPoint(pt.id)}
                    onMouseEnter={() => onHoverPoints && onHoverPoints([pt.id])}
                    onMouseLeave={() => onLeavePoints && onLeavePoints()}
                  >
                    <div className="point-triage-top">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span
                          className="point-badge"
                          style={{ background: getPointColor(pt.id), color: '#ffffff' }}
                        >
                          #{pt.id}
                        </span>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                            {pt.name}
                          </div>
                          <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>
                            {pt.skeletonLabel} &middot; {pt.isAnchor ? '⭐ Điểm Neo (≤3% IOD)' : 'Điểm Viền (≤5% IOD)'}
                          </div>
                        </div>
                      </div>

                      <span
                        className={`source-badge ${pt.currentSource === 'model' ? 'badge-ai' : 'badge-human'}`}
                      >
                        {pt.currentSource === 'model' ? '🤖 Dùng AI' : '👤 Giữ Người'}
                      </span>
                    </div>

                    {/* Coordinates & Delta */}
                    <div className="point-coords-grid">
                      <div className="coord-cell">
                        <span className="coord-label">Người (H):</span>
                        <span className="coord-val">{hCoord}</span>
                      </div>
                      <div className="coord-cell">
                        <span className="coord-label">Model (AI):</span>
                        <span className="coord-val">{mCoord}</span>
                      </div>
                      <div className="coord-cell delta">
                        <span className="coord-label">Độ lệch:</span>
                        <span
                          className={`delta-val ${pt.isExceeded ? 'exceeded' : 'passed'}`}
                        >
                          {pt.distPx !== null ? `${pt.distPx.toFixed(1)}px (${pt.pctIod?.toFixed(1)}%)` : '--'}
                        </span>
                      </div>
                    </div>

                    {/* 2-Button Choice Segment */}
                    <div className="point-triage-actions">
                      <button
                        type="button"
                        className={`triage-btn triage-btn-human ${pt.currentSource === 'human' ? 'active' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSetPointSource && onSetPointSource(pt.id, 'human');
                        }}
                      >
                        👤 Giữ Nhãn Người
                      </button>
                      <button
                        type="button"
                        className={`triage-btn triage-btn-model ${pt.currentSource === 'model' ? 'active' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSetPointSource && onSetPointSource(pt.id, 'model');
                        }}
                      >
                        🤖 Lấy Tọa Độ AI
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </aside>
  );
};
