import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { FrameDetail, Keypoint } from '../types';
import {
  SKELETONS,
  ANCHOR_POINTS,
  POINT_ANATOMY_NAMES,
  getPointColor,
} from '../vf50_constants';
import { getImageUrl } from '../api';

interface CanvasViewerProps {
  frame: FrameDetail | null;
  sessionId: string;
  showHuman: boolean;
  setShowHuman: (v: boolean) => void;
  showModel: boolean;
  setShowModel: (v: boolean) => void;
  showLabels: boolean;
  setShowLabels: (v: boolean) => void;
  showVectors: boolean;
  setShowVectors: (v: boolean) => void;
  showAnchors: boolean;
  setShowAnchors: (v: boolean) => void;
  highlightedPointIds?: Set<number>;
  selectedPointId?: number | null;
  onSelectPoint?: (pointId: number | null) => void;
  onSetPointSource?: (pointId: number, source: 'human' | 'model') => void;
}

export const CanvasViewer: React.FC<CanvasViewerProps> = ({
  frame,
  sessionId,
  showHuman,
  setShowHuman,
  showModel,
  setShowModel,
  showLabels,
  setShowLabels,
  showVectors,
  setShowVectors,
  showAnchors,
  setShowAnchors,
  highlightedPointIds,
  selectedPointId,
  onSelectPoint,
  onSetPointSource,
}) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hasDragged, setHasDragged] = useState<boolean>(false);
  const [hoveredPoint, setHoveredPoint] = useState<Keypoint | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number }>({ w: 1280, h: 720 });

  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-center & fit the image to the exact middle of the viewport
  const fitToCenter = useCallback((targetW = imgDims.w, targetH = imgDims.h) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const pad = 36;
    const scale = Math.min((rect.width - pad * 2) / targetW, (rect.height - pad * 2) / targetH);
    const clampedScale = Math.max(0.15, Math.min(scale, 3.0));
    const panX = (rect.width - targetW * clampedScale) / 2;
    const panY = (rect.height - targetH * clampedScale) / 2;

    setZoom(clampedScale);
    setPan({ x: panX, y: panY });
  }, [imgDims]);

  // Recenter whenever a new frame loads
  useEffect(() => {
    if (frame) {
      fitToCenter();
    }
  }, [frame?.id, frame?.frame_index, fitToCenter]);

  // Zoom anchored at a specific focal point (fx, fy), preserving that point's screen position
  const zoomAroundPoint = useCallback((fx: number, fy: number, factor: number) => {
    setZoom((prevZoom) => {
      const nextZoom = Math.min(6.0, Math.max(0.2, prevZoom * factor));
      const ratio = nextZoom / prevZoom;

      setPan((prevPan) => ({
        x: fx - (fx - prevPan.x) * ratio,
        y: fy - (fy - prevPan.y) * ratio,
      }));

      return nextZoom;
    });
  }, []);

  // Mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const delta = Math.max(-100, Math.min(100, e.deltaY));
    const zoomFactor = Math.exp(-delta * 0.00045);

    const focalX = mouseX * 0.65 + centerX * 0.35;
    const focalY = mouseY * 0.65 + centerY * 0.35;

    zoomAroundPoint(focalX, focalY, zoomFactor);
  };

  const handleZoomIn = () => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    zoomAroundPoint(rect.width / 2, rect.height / 2, 1.12);
  };

  const handleZoomOut = () => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    zoomAroundPoint(rect.width / 2, rect.height / 2, 0.89);
  };

  // Drag pan controls
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setHasDragged(false);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const newPanX = e.clientX - dragStart.x;
      const newPanY = e.clientY - dragStart.y;
      if (Math.hypot(newPanX - pan.x, newPanY - pan.y) > 3) {
        setHasDragged(true);
      }
      setPan({ x: newPanX, y: newPanY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const humanKpts = frame?.human_keypoints || {};
  const modelKpts = frame?.model_keypoints || {};
  const pointSources = frame?.point_sources || {};

  // Build SVG polygon/polyline path from a range of points
  const buildSkeletonPath = (
    kpts: Record<string, Keypoint>,
    range: [number, number],
    closed: boolean
  ): string => {
    const coords: [number, number][] = [];
    for (let i = range[0]; i <= range[1]; i++) {
      const pt = kpts[i] || kpts[String(i)];
      if (pt && pt.x !== null && pt.y !== null && pt.state !== 'outside') {
        coords.push([pt.x, pt.y]);
      }
    }
    if (coords.length < 2) return '';
    const d = coords.map((c, idx) => `${idx === 0 ? 'M' : 'L'} ${c[0]} ${c[1]}`).join(' ');
    return closed ? `${d} Z` : d;
  };

  // Collect violation point IDs for prominent highlight
  const violationPointIds = useMemo(() => {
    const s = new Set<number>();
    if (!frame?.rule_violations) return s;
    for (const v of frame.rule_violations) {
      if (v.points) {
        for (const p of v.points) {
          s.add(p);
        }
      }
    }
    return s;
  }, [frame?.rule_violations]);

  // Selected Point Object
  const selectedPointObj = useMemo(() => {
    if (selectedPointId === null || selectedPointId === undefined) return null;
    const h = humanKpts[selectedPointId] || humanKpts[String(selectedPointId)];
    const m = modelKpts[selectedPointId] || modelKpts[String(selectedPointId)];
    const isAnchor = ANCHOR_POINTS.has(selectedPointId);
    const maxTolPct = isAnchor ? 3.0 : 5.0;
    let distPx: number | null = null;
    let pctIod: number | null = null;
    let isExceeded = false;

    if (h && m && h.x !== null && h.y !== null && m.x !== null && m.y !== null) {
      distPx = Math.hypot(h.x - m.x, h.y - m.y);
      if (frame?.iod && frame.iod > 0) {
        pctIod = (distPx / frame.iod) * 100;
        isExceeded = pctIod > maxTolPct;
      }
    }

    const currentSource = pointSources[String(selectedPointId)] || pointSources[selectedPointId] || 'human';

    return {
      id: selectedPointId,
      name: POINT_ANATOMY_NAMES[selectedPointId] || `Điểm #${selectedPointId}`,
      isAnchor,
      distPx,
      pctIod,
      isExceeded,
      currentSource,
      h,
      m,
    };
  }, [selectedPointId, humanKpts, modelKpts, pointSources, frame?.iod]);

  if (!frame) {
    return (
      <div className="canvas-wrapper">
        <div className="canvas-empty-state">
          <div style={{ fontSize: '28px', color: 'var(--text-muted)' }}>🎯</div>
          <div style={{ fontWeight: 600 }}>Chưa chọn khung hình kiểm tra</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Vui lòng nhấp chọn một khung hình từ danh sách bên trái
          </div>
        </div>
      </div>
    );
  }

  const imageUrl = getImageUrl(sessionId, frame.image_filename);

  // Scaled stroke widths
  const ptRadius = Math.max(1.8, Math.min(3.6, 2.5 / zoom));
  const anchorRadius = Math.max(2.4, Math.min(4.8, 3.5 / zoom));
  const modelRadius = Math.max(1.4, Math.min(2.8, 1.8 / zoom));
  const violationRadius = Math.max(3.8, Math.min(8.0, 5.0 / zoom));
  const strokeWidthLine = Math.max(0.7, Math.min(1.8, 1.1 / zoom));
  const labelFontSize = Math.max(7, Math.min(12, 9 / zoom));

  return (
    <div className="canvas-wrapper">
      {/* Top Floating Glass Toolbar */}
      <div className="canvas-toolbar">
        <div className="canvas-layer-toggles">
          <button
            type="button"
            className={`layer-toggle-btn ${showHuman ? 'active' : ''}`}
            onClick={() => setShowHuman(!showHuman)}
            title="Bật/tắt nhãn người gán tay (Màu xanh lá)"
          >
            <span>👁</span> Gán tay
          </button>
          <button
            type="button"
            className={`layer-toggle-btn ${showModel ? 'active' : ''}`}
            onClick={() => setShowModel(!showModel)}
            title="Bật/tắt điểm dự đoán tham chiếu AI (MediaPipe)"
          >
            <span>🤖</span> Model AI
          </button>
          <button
            type="button"
            className={`layer-toggle-btn ${showVectors ? 'active' : ''}`}
            onClick={() => setShowVectors(!showVectors)}
            title="Bật/tắt đường véc-tơ lệch tọa độ giữa người và AI"
          >
            <span>↗</span> Vector lệch
          </button>
          <button
            type="button"
            className={`layer-toggle-btn ${showAnchors ? 'active' : ''}`}
            onClick={() => setShowAnchors(!showAnchors)}
            title="Bật/tắt làm nổi bật 12 điểm neo giải phẫu"
          >
            <span>⭐</span> 12 Điểm neo
          </button>
          <button
            type="button"
            className={`layer-toggle-btn ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels(!showLabels)}
            title="Bật/tắt số thứ tự điểm (0 - 49)"
          >
            <span>#</span> Số điểm
          </button>
        </div>

        {/* Zoom & Fit Controls */}
        <div className="canvas-controls-group">
          <button type="button" className="btn-outline" onClick={handleZoomOut} title="Thu nhỏ (-)" style={{ padding: '3px 8px' }}>
            -
          </button>
          <span style={{ fontSize: '0.74rem', fontFamily: 'var(--font-mono)', minWidth: '42px', textAlign: 'center' }}>
            {Math.round(zoom * 100)}%
          </span>
          <button type="button" className="btn-outline" onClick={handleZoomIn} title="Phóng to (+)" style={{ padding: '3px 8px' }}>
            +
          </button>
          <button
            type="button"
            className="btn-outline"
            onClick={() => fitToCenter()}
            title="Tự động căn giữa và vừa vặn màn hình"
            style={{ padding: '3px 8px', fontSize: '0.74rem' }}
          >
            ⛶ Căn giữa
          </button>
        </div>
      </div>

      {/* Main Viewport Stage */}
      <div
        ref={containerRef}
        className="canvas-screen"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={() => {
          if (!hasDragged && onSelectPoint) {
            onSelectPoint(null);
          }
        }}
      >
        <div
          className="canvas-stage"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        >
          {/* Base Face Image */}
          <img
            src={imageUrl}
            alt={frame.image_filename}
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
                fitToCenter(img.naturalWidth, img.naturalHeight);
              }
            }}
            style={{
              display: 'block',
              width: `${imgDims.w}px`,
              height: `${imgDims.h}px`,
              maxWidth: 'none',
              pointerEvents: 'none',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)',
              border: '1px solid #dcdfe6',
              borderRadius: '2px',
            }}
          />

          {/* SVG Keypoint and Vector Overlays */}
          <svg
            viewBox={`0 0 ${imgDims.w} ${imgDims.h}`}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: `${imgDims.w}px`,
              height: `${imgDims.h}px`,
              pointerEvents: 'none',
            }}
          >
            {/* 1. Error Vector Lines (Human -> Model) */}
            {showVectors &&
              Object.keys(humanKpts).map((key) => {
                const hPt = humanKpts[key];
                const mPt = modelKpts[key];
                if (!hPt || !mPt || hPt.x === null || hPt.y === null || mPt.x === null || mPt.y === null) return null;
                const dist = Math.hypot(hPt.x - mPt.x, hPt.y - mPt.y);
                if (dist < 1.0) return null;

                const isViolated = violationPointIds.has(hPt.id) || highlightedPointIds?.has(hPt.id);

                return (
                  <line
                    key={`vec_${key}`}
                    x1={hPt.x}
                    y1={hPt.y}
                    x2={mPt.x}
                    y2={mPt.y}
                    stroke={isViolated ? '#EF4444' : '#F59E0B'}
                    strokeWidth={isViolated ? strokeWidthLine * 1.5 : strokeWidthLine}
                    strokeDasharray={isViolated ? 'none' : '2,2'}
                    opacity={0.9}
                  />
                );
              })}

            {/* 2. Model AI Predictions (Cyan Skeletons & Points) */}
            {showModel &&
              SKELETONS.map((sk) => {
                const pathStr = buildSkeletonPath(modelKpts, sk.pointRange, sk.closed);
                if (!pathStr) return null;
                return (
                  <path
                    key={`model_sk_${sk.name}`}
                    d={pathStr}
                    fill="none"
                    stroke="#0284C7"
                    strokeWidth={strokeWidthLine * 0.8}
                    strokeDasharray="3,3"
                    opacity={0.7}
                  />
                );
              })}

            {showModel &&
              Object.values(modelKpts).map((pt) => {
                if (pt.x === null || pt.y === null || pt.state === 'outside') return null;
                return (
                  <circle
                    key={`model_pt_${pt.id}`}
                    cx={pt.x}
                    cy={pt.y}
                    r={modelRadius}
                    fill="#0284C7"
                    opacity={0.8}
                  />
                );
              })}

            {/* 3. Human Annotated Labels (Skeletons & Keypoints) */}
            {showHuman &&
              SKELETONS.map((sk) => {
                const pathStr = buildSkeletonPath(humanKpts, sk.pointRange, sk.closed);
                if (!pathStr) return null;
                return (
                  <path
                    key={`human_sk_${sk.name}`}
                    d={pathStr}
                    fill="none"
                    stroke={sk.color || '#10B981'}
                    strokeWidth={strokeWidthLine}
                    opacity={0.92}
                  />
                );
              })}

            {showHuman &&
              Object.values(humanKpts).map((pt) => {
                if (pt.x === null || pt.y === null || pt.state === 'outside') return null;
                const isAnchor = showAnchors && ANCHOR_POINTS.has(pt.id);
                const isViolated = violationPointIds.has(pt.id);
                const isHighlighted = highlightedPointIds?.has(pt.id);
                const isSelected = selectedPointId === pt.id;
                const isModelSource = pointSources[String(pt.id)] === 'model';

                return (
                  <g
                    key={`human_pt_${pt.id}`}
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onSelectPoint) {
                        onSelectPoint(pt.id);
                      }
                    }}
                    onMouseEnter={() => setHoveredPoint(pt)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  >
                    {/* Ring for Selected Point */}
                    {isSelected && (
                      <circle
                        cx={pt.x}
                        cy={pt.y}
                        r={violationRadius * 1.8}
                        fill="none"
                        stroke="#2563EB"
                        strokeWidth={strokeWidthLine * 2.2}
                        strokeDasharray="4,2"
                      >
                        <animateTransform
                          attributeName="transform"
                          type="rotate"
                          from={`0 ${pt.x} ${pt.y}`}
                          to={`360 ${pt.x} ${pt.y}`}
                          dur="6s"
                          repeatCount="indefinite"
                        />
                      </circle>
                    )}

                    {/* Pulsing ring for violation / highlighted point */}
                    {(isViolated || isHighlighted) && !isSelected && (
                      <circle
                        cx={pt.x}
                        cy={pt.y}
                        r={violationRadius}
                        fill="none"
                        stroke="#EF4444"
                        strokeWidth={strokeWidthLine * 1.5}
                        opacity={isHighlighted ? 1 : 0.85}
                      >
                        <animate
                          attributeName="r"
                          values={`${violationRadius};${violationRadius * 1.6};${violationRadius}`}
                          dur="1.8s"
                          repeatCount="indefinite"
                        />
                      </circle>
                    )}

                    {/* Point circle */}
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={isAnchor ? anchorRadius : ptRadius}
                      fill={isViolated ? '#EF4444' : isModelSource ? '#0284C7' : getPointColor(pt.id)}
                      stroke={isSelected ? '#2563EB' : isAnchor ? '#FBBF24' : '#FFFFFF'}
                      strokeWidth={isSelected ? 2.0 : isAnchor ? 1.6 : 0.75}
                    />

                    {/* Point Label Number */}
                    {showLabels && (
                      <text
                        x={pt.x + ptRadius + 1}
                        y={pt.y - ptRadius - 1}
                        fontSize={labelFontSize}
                        fill="#FFFFFF"
                        fontWeight="bold"
                        stroke="#000000"
                        strokeWidth={0.6}
                      >
                        {pt.id}
                      </text>
                    )}
                  </g>
                );
              })}
          </svg>
        </div>

        {/* Hover Tooltip (When NOT selecting a point) */}
        {!selectedPointId && hoveredPoint && hoveredPoint.x !== null && hoveredPoint.y !== null && (
          <div
            style={{
              position: 'absolute',
              left: pan.x + hoveredPoint.x * zoom + 12,
              top: pan.y + hoveredPoint.y * zoom - 24,
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--border-strong)',
              borderRadius: '6px',
              padding: '5px 9px',
              fontSize: '11px',
              color: 'white',
              boxShadow: 'var(--shadow-md)',
              pointerEvents: 'none',
              zIndex: 100,
            }}
          >
            {(() => {
              const mkp = frame?.model_keypoints?.[hoveredPoint.id];
              let distPx: number | null = null;
              let normPct: string | null = null;
              const isAnchor = ANCHOR_POINTS.has(hoveredPoint.id);
              const maxTolPct = isAnchor ? 3.0 : 5.0;

              if (mkp && mkp.x !== null && mkp.y !== null && hoveredPoint.x !== null && hoveredPoint.y !== null) {
                distPx = Math.hypot(hoveredPoint.x - mkp.x, hoveredPoint.y - mkp.y);
                if (frame?.iod && frame.iod > 0) {
                  normPct = ((distPx / frame.iod) * 100).toFixed(1);
                }
              }

              const isExceeded = normPct !== null && parseFloat(normPct) > maxTolPct;
              const isModel = pointSources[String(hoveredPoint.id)] === 'model';

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <div style={{ fontWeight: 700, color: getPointColor(hoveredPoint.id), fontSize: '11.5px' }}>
                    #{hoveredPoint.id}: {POINT_ANATOMY_NAMES[hoveredPoint.id] || ''} {isAnchor ? '⭐ (Neo)' : ''}
                  </div>
                  <div style={{ color: '#CBD5E1', fontSize: '10.5px' }}>
                    Nguồn: <strong style={{ color: isModel ? '#38BDF8' : '#34D399' }}>{isModel ? '🤖 Model AI' : '👤 Người'}</strong>
                  </div>
                  {distPx !== null && (
                    <div
                      style={{
                        color: isExceeded ? '#F87171' : '#34D399',
                        fontSize: '10.5px',
                        borderTop: '1px solid rgba(255,255,255,0.1)',
                        paddingTop: '2px',
                        marginTop: '2px',
                        fontWeight: 600,
                      }}
                    >
                      Lệch: {distPx.toFixed(1)}px ({normPct}% IOD) {isExceeded ? '⚠️ Vượt dung sai' : '✓ Chuẩn'}
                    </div>
                  )}
                  <div style={{ fontSize: '9.5px', color: '#94A3B8', marginTop: '1px' }}>
                    💡 Click điểm để chọn nhanh nguồn Người / AI
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* Interactive Floating Point Popover when a point is SELECTED */}
        {selectedPointObj && selectedPointObj.h && selectedPointObj.h.x !== null && selectedPointObj.h.y !== null && (
          <div
            className="canvas-point-popover"
            style={{
              position: 'absolute',
              left: Math.max(16, Math.min(pan.x + selectedPointObj.h.x * zoom + 16, (containerRef.current?.clientWidth || 800) - 270)),
              top: Math.max(50, Math.min(pan.y + selectedPointObj.h.y * zoom - 65, (containerRef.current?.clientHeight || 600) - 170)),
              zIndex: 110,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Popover Header */}
            <div className="popover-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span
                  className="point-badge"
                  style={{ background: getPointColor(selectedPointObj.id), color: '#ffffff' }}
                >
                  #{selectedPointObj.id}
                </span>
                <span style={{ fontWeight: 700, fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                  {selectedPointObj.name}
                </span>
              </div>
              <button
                type="button"
                className="btn-popover-close"
                onClick={() => onSelectPoint && onSelectPoint(null)}
                title="Đóng bảng chọn điểm"
              >
                ✕
              </button>
            </div>

            {/* Popover Details */}
            <div className="popover-body">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                <span>Đang dùng nguồn:</span>
                <strong style={{ color: selectedPointObj.currentSource === 'model' ? 'var(--cyan)' : 'var(--success)' }}>
                  {selectedPointObj.currentSource === 'model' ? '🤖 Model AI' : '👤 Nhãn Người'}
                </strong>
              </div>

              {selectedPointObj.distPx !== null && (
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                  <span>Sai lệch so với AI:</span>
                  <span style={{ color: selectedPointObj.isExceeded ? 'var(--danger)' : 'var(--success)', fontWeight: 700 }}>
                    {selectedPointObj.distPx.toFixed(1)}px ({selectedPointObj.pctIod?.toFixed(1)}% IOD)
                  </span>
                </div>
              )}

              {/* Action Buttons: 👤 Giữ Người vs 🤖 Dùng AI */}
              <div className="popover-actions">
                <button
                  type="button"
                  className={`btn-popover-choice human ${selectedPointObj.currentSource === 'human' ? 'active' : ''}`}
                  onClick={() => onSetPointSource && onSetPointSource(selectedPointObj.id, 'human')}
                  title="Chọn giữ tọa độ dán nhãn của Người"
                >
                  👤 Giữ Người
                </button>
                <button
                  type="button"
                  className={`btn-popover-choice model ${selectedPointObj.currentSource === 'model' ? 'active' : ''}`}
                  onClick={() => onSetPointSource && onSetPointSource(selectedPointObj.id, 'model')}
                  title="Chọn lấy tọa độ gợi ý của Model AI"
                >
                  🤖 Dùng AI
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Non-obstructive Floating Bottom Legend with 7 CVAT Skeletons */}
        <div className="canvas-bottom-legend">
          {SKELETONS.map((sk) => (
            <div key={`legend_sk_${sk.name}`} className="legend-chip" title={`${sk.label} (${sk.pointRange[0]}-${sk.pointRange[1]})`}>
              <span className="dot-indicator" style={{ background: sk.color }} />
              <span>{sk.label}</span>
            </div>
          ))}
          <div className="legend-chip" title="Điểm ước lượng từ MediaPipe Face Mesh">
            <span className="dot-indicator" style={{ background: '#0284C7' }} />
            <span>Model AI</span>
          </div>
          <div className="legend-chip" title="Điểm vi phạm luật hoặc vượt dung sai">
            <span className="dot-indicator" style={{ background: '#EF4444' }} />
            <span>Lỗi vi phạm</span>
          </div>
        </div>
      </div>
    </div>
  );
};
