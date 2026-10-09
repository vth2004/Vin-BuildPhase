import React, { useState, useEffect, useCallback, useMemo } from 'react';
import type {
  FrameDetail,
  FrameListItem,
  SessionInfo,
} from './types';
import * as api from './api';
import {
  checkBackendHealth,
  createDemoSession,
  getFrameDetail,
  getSessionFrames,
  getSessions,
  setFrameStatus,
} from './api';
import { Header } from './components/Header';
import { FrameList } from './components/FrameList';
import { CanvasViewer } from './components/CanvasViewer';
import { Inspector } from './components/Inspector';
import { UploadModal } from './components/UploadModal';
import { GuidelineModal } from './components/GuidelineModal';

export const App: React.FC = () => {
  // Session State
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  // Frames List State
  const [frames, setFrames] = useState<FrameListItem[]>([]);
  const [totalFrames, setTotalFrames] = useState<number>(0);
  const [sortBy, setSortBy] = useState<string>('severity');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'suspect' | 'severe'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'reviewed'>('all');
  const [searchImage, setSearchImage] = useState<string>('');

  // Active Frame State
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null);
  const [currentFrameDetail, setCurrentFrameDetail] = useState<FrameDetail | null>(null);

  // Layer Toggles
  const [showHuman, setShowHuman] = useState<boolean>(true);
  const [showModel, setShowModel] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(false);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [showAnchors, setShowAnchors] = useState<boolean>(true);

  // Hovered & Selected keypoints
  const [highlightedPoints, setHighlightedPoints] = useState<Set<number> | undefined>(undefined);
  const [selectedPointId, setSelectedPointId] = useState<number | null>(null);

  // Modals & Status
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [isDemoLoading, setIsDemoLoading] = useState<boolean>(false);
  const [backendHealthy, setBackendHealthy] = useState<boolean>(true);
  const [feedbackMsg, setFeedbackMsg] = useState<string>('');
  const [isGuidelineOpen, setIsGuidelineOpen] = useState<boolean>(false);
  const [selectedGuidelineCode, setSelectedGuidelineCode] = useState<string | null>(null);

  // 1. Initial Health Check & Session Load
  const loadSessions = useCallback(async (selectId?: string) => {
    try {
      const sessList = await getSessions();
      setSessions(sessList);
      if (sessList.length > 0) {
        const toSelect = selectId || sessList[0].session_id;
        setSelectedSessionId(toSelect);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }, []);

  useEffect(() => {
    checkBackendHealth()
      .then(() => setBackendHealthy(true))
      .catch(() => setBackendHealthy(false));

    loadSessions();
  }, [loadSessions]);

  // 2. Load Frames when Session or Backend Filters Change
  const loadFrames = useCallback(async () => {
    if (!selectedSessionId) return;
    try {
      const res = await getSessionFrames(selectedSessionId, 1, 100, sortBy, 'all');
      setFrames(res.items);
      setTotalFrames(res.total);
      if (res.items.length > 0) {
        const urlParams = new URLSearchParams(window.location.search);
        const frameParam = urlParams.get('frame');
        const targetIdx = frameParam !== null ? parseInt(frameParam, 10) : NaN;
        const matched = !isNaN(targetIdx) ? res.items.find((it) => it.frame_index === targetIdx) : null;
        setSelectedFrameIndex(matched ? matched.frame_index : res.items[0].frame_index);
      } else {
        setSelectedFrameIndex(null);
        setCurrentFrameDetail(null);
      }
    } catch (err) {
      console.error('Failed to load frames:', err);
    }
  }, [selectedSessionId, sortBy]);

  useEffect(() => {
    loadFrames();
  }, [loadFrames]);

  // 3. Filter Frames Locally (Severity, Status, Search)
  const filteredFrames = useMemo(() => {
    return frames.filter((f) => {
      // Severity Filter
      if (severityFilter === 'suspect' && f.nme < 0.02 && f.error_count === 0) return false;
      if (severityFilter === 'severe' && f.nme < 0.035 && f.error_count === 0) return false;

      // Status Filter
      if (statusFilter === 'pending' && f.status !== 'pending') return false;
      if (statusFilter === 'reviewed' && f.status === 'pending') return false;

      // Search Filter
      if (searchImage && !f.image_filename.toLowerCase().includes(searchImage.toLowerCase())) {
        return false;
      }

      return true;
    });
  }, [frames, severityFilter, statusFilter, searchImage]);

  // Keep selected frame valid after filtering
  useEffect(() => {
    if (filteredFrames.length > 0) {
      const stillInList = filteredFrames.some((f) => f.frame_index === selectedFrameIndex);
      if (!stillInList) {
        setSelectedFrameIndex(filteredFrames[0].frame_index);
      }
    } else {
      setSelectedFrameIndex(null);
      setCurrentFrameDetail(null);
    }
  }, [filteredFrames, selectedFrameIndex]);

  // 4. Load Frame Detail when Selected Frame Index changes
  useEffect(() => {
    setSelectedPointId(null);
    if (!selectedSessionId || selectedFrameIndex === null) {
      setCurrentFrameDetail(null);
      return;
    }

    getFrameDetail(selectedSessionId, selectedFrameIndex)
      .then((detail) => setCurrentFrameDetail(detail))
      .catch((err) => console.error('Failed to fetch frame detail:', err));
  }, [selectedSessionId, selectedFrameIndex]);

  // 5. Handlers
  const handleRunDemo = async () => {
    setIsDemoLoading(true);
    try {
      const demo = await createDemoSession();
      await loadSessions(demo.session_id);
      setIsDemoLoading(false);
      setFeedbackMsg('Đã tạo thành công phiên bản mẫu thử nghiệm!');
      setTimeout(() => setFeedbackMsg(''), 4000);
    } catch (err) {
      console.error('Failed to create demo:', err);
      setIsDemoLoading(false);
    }
  };

  const handleStatusChange = async (frameIndex: number, newStatus: 'pending' | 'reviewed' | 'fixed') => {
    if (!selectedSessionId) return;
    try {
      await setFrameStatus(selectedSessionId, frameIndex, newStatus);
      setFrames((prev) =>
        prev.map((f) => (f.frame_index === frameIndex ? { ...f, status: newStatus } : f))
      );
      if (currentFrameDetail && currentFrameDetail.frame_index === frameIndex) {
        setCurrentFrameDetail({ ...currentFrameDetail, status: newStatus });
      }
    } catch (err) {
      console.error('Failed to update status:', err);
    }
  };

  const handleApplyModelFix = async (pointIds?: number[]) => {
    if (!selectedSessionId || selectedFrameIndex === null) return;
    try {
      const updated = await api.applyModelFix(selectedSessionId, selectedFrameIndex, pointIds);
      setCurrentFrameDetail(updated);
      setFrames((prev) =>
        prev.map((f) =>
          f.frame_index === selectedFrameIndex
            ? {
                ...f,
                status: updated.status,
                nme: updated.nme,
                iod: updated.iod,
                error_count: updated.error_count,
                severity_score: updated.severity_score,
                rule_violations: updated.rule_violations,
                case_type: updated.case_type,
                ai_reliability: updated.ai_reliability,
                case_label: updated.case_label,
              }
            : f
        )
      );
      setFeedbackMsg(
        pointIds && pointIds.length > 0
          ? `✓ Đã sửa ${pointIds.length} điểm theo Model AI!`
          : `✓ Đã sửa tất cả lỗi Frame #${selectedFrameIndex} theo Model AI!`
      );
      setTimeout(() => setFeedbackMsg(''), 3500);
    } catch (e: any) {
      console.error('Failed to apply model fix', e);
      alert('Lỗi khi sửa theo Model AI: ' + (e?.message || 'Không xác định'));
    }
  };

  const handleSetPointSource = async (pointId: number, source: 'human' | 'model') => {
    if (!selectedSessionId || selectedFrameIndex === null) return;
    try {
      const updated = await api.setPointSource(selectedSessionId, selectedFrameIndex, pointId, source);
      setCurrentFrameDetail(updated);
      setFrames((prev) =>
        prev.map((f) =>
          f.frame_index === selectedFrameIndex
            ? {
                ...f,
                status: updated.status,
                nme: updated.nme,
                iod: updated.iod,
                error_count: updated.error_count,
                severity_score: updated.severity_score,
                rule_violations: updated.rule_violations,
                case_type: updated.case_type,
                ai_reliability: updated.ai_reliability,
                case_label: updated.case_label,
              }
            : f
        )
      );
      setFeedbackMsg(
        source === 'model'
          ? `✓ Điểm #${pointId} đã chuyển sang dùng tọa độ Model AI!`
          : `✓ Điểm #${pointId} đã khôi phục về nhãn Người!`
      );
      setTimeout(() => setFeedbackMsg(''), 3000);
    } catch (e: any) {
      console.error('Failed to set point source', e);
      alert('Lỗi khi đổi nguồn điểm: ' + (e?.message || 'Không xác định'));
    }
  };

  // 6. Navigation
  const handlePrevFrame = () => {
    if (filteredFrames.length === 0 || selectedFrameIndex === null) return;
    const currIdx = filteredFrames.findIndex((f) => f.frame_index === selectedFrameIndex);
    if (currIdx > 0) {
      setSelectedFrameIndex(filteredFrames[currIdx - 1].frame_index);
    }
  };

  const handleNextFrame = () => {
    if (filteredFrames.length === 0 || selectedFrameIndex === null) return;
    const currIdx = filteredFrames.findIndex((f) => f.frame_index === selectedFrameIndex);
    if (currIdx >= 0 && currIdx < filteredFrames.length - 1) {
      setSelectedFrameIndex(filteredFrames[currIdx + 1].frame_index);
    }
  };

  // 7. Keyboard Shortcuts (K: Giữ nhãn, M: Lấy Model AI, S: Bỏ qua, ArrowDown, ArrowUp)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;
      if (selectedFrameIndex === null) return;

      const key = e.key.toLowerCase();
      if (key === 'k') {
        handleStatusChange(selectedFrameIndex, 'fixed');
      } else if (key === 'm') {
        handleApplyModelFix();
      } else if (key === 's') {
        handleStatusChange(selectedFrameIndex, 'pending');
      } else if (e.key === 'ArrowDown' || key === 'j') {
        e.preventDefault();
        handleNextFrame();
      } else if (e.key === 'ArrowUp' || key === 'u') {
        e.preventDefault();
        handlePrevFrame();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedFrameIndex, filteredFrames, selectedSessionId]);

  // 8. Guideline Handbook & Batch Auto-Fix Handlers
  const handleOpenGuideline = (ruleCode?: string) => {
    setSelectedGuidelineCode(ruleCode || null);
    setIsGuidelineOpen(true);
  };

  return (
    <div className="app-container">
      {/* 1. Main Header */}
      <Header
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        onSelectSession={(id) => setSelectedSessionId(id)}
        onOpenUpload={() => setIsUploadOpen(true)}
        onRunDemo={handleRunDemo}
        isDemoLoading={isDemoLoading}
        backendHealthy={backendHealthy}
        onOpenGuideline={() => handleOpenGuideline('R01')}
      />

      {/* 2. Sub-Toolbar: Compact Filters Bar & Shortcuts */}
      {/* 2. Sub-Toolbar: Balanced Filters Bar (Clean, Balanced & Redundancy-Free) */}
      <div className="app-toolbar">
        <div className="toolbar-left">
          {/* Mức độ NME (Segmented Control) */}
          <div className="toolbar-group">
            <span className="toolbar-label">Mức độ:</span>
            <div className="segmented-filter">
              <button
                className={`segmented-filter-btn ${severityFilter === 'all' ? 'active' : ''}`}
                onClick={() => setSeverityFilter('all')}
              >
                Tất cả
              </button>
              <button
                className={`segmented-filter-btn ${severityFilter === 'suspect' ? 'active' : ''}`}
                onClick={() => setSeverityFilter('suspect')}
              >
                Nghi ngờ (≥2%)
              </button>
              <button
                className={`segmented-filter-btn ${severityFilter === 'severe' ? 'active' : ''}`}
                onClick={() => setSeverityFilter('severe')}
              >
                Lỗi nặng (&gt; 3.5%)
              </button>
            </div>
          </div>

          <div className="toolbar-divider" />

          {/* Trạng thái duyệt */}
          <div className="toolbar-group">
            <span className="toolbar-label">Trạng thái:</span>
            <select
              className="toolbar-select"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
            >
              <option value="all">Tất cả trạng thái</option>
              <option value="pending">Chưa sửa</option>
              <option value="reviewed">Đã rà soát / Đã sửa</option>
            </select>
          </div>

          <div className="toolbar-divider" />

          {/* Sắp xếp danh sách */}
          <div className="toolbar-group">
            <span className="toolbar-label">Sắp xếp:</span>
            <select
              className="toolbar-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="severity">Ưu tiên lỗi nặng</option>
              <option value="frame_index">Thứ tự Frame (0 - N)</option>
              <option value="nme">NME cao nhất</option>
            </select>
          </div>
        </div>

        {/* Toolbar Right: Thông báo & Hướng dẫn phím tắt góc phải */}
        <div className="toolbar-right">
          {feedbackMsg && (
            <span style={{ color: 'var(--success)', fontWeight: 600, fontSize: '0.74rem' }}>{feedbackMsg}</span>
          )}
          <div className="toolbar-shortcuts">
            <span className="toolbar-label">Phím tắt:</span>
            <kbd>K</kbd> <span className="shortcut-action">Đã sửa</span>
            <span className="shortcut-pipe">|</span>
            <kbd>M</kbd> <span className="shortcut-action">Lấy Model</span>
            <span className="shortcut-pipe">|</span>
            <kbd>S</kbd> <span className="shortcut-action">Bỏ qua</span>
            <span className="shortcut-pipe">|</span>
            <kbd>↓ / ↑</kbd> <span className="shortcut-action">Đổi ảnh</span>
          </div>
        </div>
      </div>

      {/* 3. Studio 3-Panel Workspace (No Page Scroll - Fixed Height) */}
      <div className="studio-workspace">
        {/* Panel 1 (Left): Frame List with Search & Status */}
        <FrameList
          frames={filteredFrames}
          totalFrames={totalFrames}
          selectedFrameIndex={selectedFrameIndex}
          searchImage={searchImage}
          onSearchChange={setSearchImage}
          onSelectFrame={(idx) => setSelectedFrameIndex(idx)}
          onStatusChange={handleStatusChange}
        />

        {/* Panel 2 (Center): Centered Canvas Viewport */}
        <CanvasViewer
          frame={currentFrameDetail}
          sessionId={selectedSessionId || ''}
          showHuman={showHuman}
          setShowHuman={setShowHuman}
          showModel={showModel}
          setShowModel={setShowModel}
          showLabels={showLabels}
          setShowLabels={setShowLabels}
          showVectors={showVectors}
          setShowVectors={setShowVectors}
          showAnchors={showAnchors}
          setShowAnchors={setShowAnchors}
          highlightedPointIds={highlightedPoints}
          selectedPointId={selectedPointId}
          onSelectPoint={setSelectedPointId}
          onSetPointSource={handleSetPointSource}
        />

        {/* Panel 3 (Right): Inspector (Metrics & CVAT Fix Guidelines) */}
        <Inspector
          frame={currentFrameDetail}
          onStatusChange={(st) => {
            if (selectedFrameIndex !== null) {
              handleStatusChange(selectedFrameIndex, st);
            }
          }}
          onApplyModelFix={handleApplyModelFix}
          onHoverPoints={(pts) => setHighlightedPoints(new Set(pts))}
          onLeavePoints={() => setHighlightedPoints(undefined)}
          onOpenGuidelineRule={(ruleCode) => handleOpenGuideline(ruleCode)}
          onSetPointSource={handleSetPointSource}
          selectedPointId={selectedPointId}
          onSelectPoint={setSelectedPointId}
        />
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={(newId) => loadSessions(newId)}
      />

      {/* Sổ Tay Quy Chuẩn Guideline Modal (14 Quy Tắc Chi Tiết) */}
      <GuidelineModal
        isOpen={isGuidelineOpen}
        onClose={() => setIsGuidelineOpen(false)}
        selectedRuleCode={selectedGuidelineCode}
      />
    </div>
  );
};

export default App;
