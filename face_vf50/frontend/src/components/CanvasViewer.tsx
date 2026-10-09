import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { FrameDetail, Keypoint } from '../types';
import { SKELETONS, ANCHOR_POINTS, getPointColor } from '../vf50_constants';
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
}) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
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

  // Mouse wheel zoom: Smooth, gentle step factor (~4.5%) with dampened focal point towards center
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Gentle zoom factor: ~4.5% per notch (substantially smoother than previous 15%)
    // Normalized for both discrete mouse wheel notches and continuous trackpads
    const delta = Math.max(-100, Math.min(100, e.deltaY));
    const zoomFactor = Math.exp(-delta * 0.00045);

    // Dampened focal point: blend 65% mouse position with 35% viewport center
    // This prevents extreme edge flinging/drift while keeping cursor-focused zoom feel
    const focalX = mouseX * 0.65 + centerX * 0.35;
    const focalY = mouseY * 0.65 + centerY * 0.35;

    zoomAroundPoint(focalX, focalY, zoomFactor);
  };

  // Button Zoom: Strictly anchored to the dead-center of the viewport with gentle 1.12 step
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
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Collect points involved in violations
  const violationPointIds = useMemo(() => {
    const ids = new Set<number>();
    if (frame?.rule_violations) {
      frame.rule_violations.forEach((v) => {
        if (v.code !== 'R02' && v.points && v.points.length > 0 && v.points.length < 25) {
          v.points.forEach((p) => ids.add(p));
        }
      });
    }
    return ids;
  }, [frame?.rule_violations]);

  if (!frame) {
    return (
      <div className="panel-center" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '32px', marginBottom: '8px' }}>🖼️</div>
          <p style={{ fontSize: '0.88rem' }}>Chọn một khung hình từ danh sách bên trái để bắt đầu kiểm định</p>
        </div>
      </div>
    );
  }

  const imageUrl = getImageUrl(sessionId, frame.image_filename);
  const humanKpts = frame.human_keypoints || {};
  const modelKpts = frame.model_keypoints || {};

  const iod = Math.max(12, frame.iod || 96);
  const ptRadius = Math.max(1.0, Math.min(4.0, iod * 0.035));
  const anchorRadius = ptRadius * 1.45;
  const violationRadius = ptRadius * 1.9;
  const strokeWidthLine = Math.max(0.7, Math.min(2.2, iod * 0.02));
  const modelRadius = ptRadius * 0.75;
  const labelFontSize = Math.max(6, Math.min(10, Math.round(iod * 0.08)));

  const buildSkeletonPath = (kpts: Record<string, Keypoint>, pointRange: [number, number], closed: boolean) => {
    const [start, end] = pointRange;
    let path = '';
    let first = true;
    for (let i = start; i <= end; i++) {
      const pt = kpts[i.toString()] || kpts[i as unknown as string];
      if (pt && pt.x !== null && pt.y !== null && pt.state !== 'outside') {
        if (first) {
          path += `M ${pt.x} ${pt.y} `;
          first = false;
        } else {
          path += `L ${pt.x} ${pt.y} `;
        }
      }
    }
    if (closed && !first) {
      path += 'Z';
    }
    return path;
  };

  return (
    <div className="panel-center">
      {/* Top Floating Control Bar */}
      <div className="canvas-top-bar">
        {/* Layer Toggles */}
        <div className="canvas-layer-toggles">
          <button
            className={`layer-toggle-btn ${showHuman ? 'active' : ''}`}
            onClick={() => setShowHuman(!showHuman)}
            title="Bật/tắt nhãn người gán tay (Màu xanh lá)"
          >
            <span>👁</span> Gán tay
          </button>
          <button
            className={`layer-toggle-btn ${showModel ? 'active' : ''}`}
            onClick={() => setShowModel(!showModel)}
            title="Bật/tắt điểm dự đoán tham chiếu AI (MediaPipe)"
          >
            <span>🤖</span> Model AI
          </button>
          <button
            className={`layer-toggle-btn ${showVectors ? 'active' : ''}`}
            onClick={() => setShowVectors(!showVectors)}
            title="Bật/tắt đường véc-tơ lệch tọa độ giữa người và AI"
          >
            <span>↗</span> Vector lệch
          </button>
          <button
            className={`layer-toggle-btn ${showAnchors ? 'active' : ''}`}
            onClick={() => setShowAnchors(!showAnchors)}
            title="Bật/tắt làm nổi bật 12 điểm neo giải phẫu"
          >
            <span>⭐</span> 12 Điểm neo
          </button>
          <button
            className={`layer-toggle-btn ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels(!showLabels)}
            title="Bật/tắt số thứ tự điểm (0 - 49)"
          >
            <span>#</span> Số điểm
          </button>
        </div>

        {/* Zoom & Fit Controls */}
        <div className="canvas-controls-group">
          <button className="btn-outline" onClick={handleZoomOut} title="Thu nhỏ (-)" style={{ padding: '3px 8px' }}>
            -
          </button>
          <span style={{ fontSize: '0.74rem', fontFamily: 'var(--font-mono)', minWidth: '42px', textAlign: 'center' }}>
            {Math.round(zoom * 100)}%
          </span>
          <button className="btn-outline" onClick={handleZoomIn} title="Phóng to (+)" style={{ padding: '3px 8px' }}>
            +
          </button>
          <button
            className="btn-outline"
            onClick={() => fitToCenter()}
            title="Tự động căn giữa và vừa vặn màn hình"
            style={{ padding: '3px 8px', fontSize: '0.74rem' }}
          >
            ⛶ Căn giữa
          </button>
        </div>
      </div>

      {/* Main Viewport Stage - Perfectly Centered on Light Studio Gray Workbench */}
      <div
        ref={containerRef}
        className="canvas-screen"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          className="canvas-stage"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        >
          {/* Base Face Image with elevation & crisp border */}
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

                return (
                  <g
                    key={`human_pt_${pt.id}`}
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onMouseEnter={() => setHoveredPoint(pt)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  >
                    {/* Pulsing ring for violation / highlighted point */}
                    {(isViolated || isHighlighted) && (
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
                      fill={isViolated ? '#EF4444' : getPointColor(pt.id)}
                      stroke={isAnchor ? '#FBBF24' : '#FFFFFF'}
                      strokeWidth={isAnchor ? 1.6 : 0.75}
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

        {/* Hover Tooltip */}
        {hoveredPoint && hoveredPoint.x !== null && hoveredPoint.y !== null && (
          <div
            style={{
              position: 'absolute',
              left: pan.x + hoveredPoint.x * zoom + 12,
              top: pan.y + hoveredPoint.y * zoom - 24,
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--border-strong)',
              borderRadius: '6px',
              padding: '4px 8px',
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

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <div style={{ fontWeight: 700, color: getPointColor(hoveredPoint.id), fontSize: '11.5px' }}>
                    Điểm #{hoveredPoint.id} {isAnchor ? '⭐ (Điểm Neo ≤3% IOD)' : '(Điểm Viền ≤5% IOD)'}
                  </div>
                  <div style={{ color: '#CBD5E1', fontSize: '10.5px' }}>
                    Tọa độ: ({hoveredPoint.x.toFixed(1)}, {hoveredPoint.y.toFixed(1)}) &middot; Trạng thái: <span style={{ color: hoveredPoint.state === 'occluded' ? '#FBBF24' : hoveredPoint.state === 'outside' ? '#F87171' : '#34D399', fontWeight: 600 }}>{hoveredPoint.state || 'visible'}</span>
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
                      Sai lệch: {distPx.toFixed(1)}px ({normPct}% IOD) {isExceeded ? '⚠️ Vượt dung sai' : '✓ Chuẩn'}
                    </div>
                  )}
                </div>
              );
            })()}
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
