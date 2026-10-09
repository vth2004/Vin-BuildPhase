import React, { useState, useRef, useEffect } from 'react';
import {
  getCvatXmlDownloadUrl,
  getJsonDownloadUrl,
  getCsvDownloadUrl,
  getChecklistDownloadUrl,
} from '../api';

interface HeaderProps {
  onOpenUpload: () => void;
  backendHealthy: boolean;
  sessions?: any[];
  selectedSessionId?: string | null;
  onSelectSession?: (id: string) => void;
  onRunDemo?: () => void;
  isDemoLoading?: boolean;
  onOpenGuideline: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenUpload,
  backendHealthy,
  selectedSessionId,
  onOpenGuideline,
}) => {
  const [showExportMenu, setShowExportMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    if (showExportMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showExportMenu]);

  return (
    <header className="app-header">
      {/* Brand */}
      <div className="brand-section">
        <div className="brand-logo-icon">VF</div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <h1 className="brand-title">Face Landmark VF-50 QA Auditor</h1>
          </div>
          <div className="brand-subtitle">Hand-Labeling Copilot &middot; Chuẩn VinFast VF50 GL v1.3</div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="header-actions">
        {/* Nút Mở Sổ Tay Quy Chuẩn Guideline (14 Quy Tắc) */}
        <button
          className="btn header-guideline-btn"
          onClick={onOpenGuideline}
          title="Xem chi tiết 14 Quy chuẩn Guideline VinFast v1.3 kèm công thức toán học và cách sửa trên CVAT"
        >
          <span className="btn-icon">📖</span> Sổ tay Quy chuẩn (14 Luật)
        </button>

        {/* Export Cleaned Annotations Dropdown */}
        {selectedSessionId && (
          <div style={{ position: 'relative' }} ref={menuRef}>
            <button
              className="btn btn-secondary"
              onClick={() => setShowExportMenu(!showExportMenu)}
              title="Xuất file nhãn đã hiệu chỉnh theo chuẩn CVAT XML hoặc JSON"
              style={{
                background: '#f8fafc',
                color: '#0f172a',
                border: '1px solid #cbd5e1',
                fontWeight: 600,
                fontSize: '0.78rem',
              }}
            >
              📥 Xuất nhãn đã sửa ▾
            </button>

            {showExportMenu && (
              <div className="export-dropdown-menu">
                <a
                  href={getCvatXmlDownloadUrl(selectedSessionId)}
                  download
                  className="export-dropdown-item"
                  onClick={() => setShowExportMenu(false)}
                >
                  <span className="export-item-icon">📄</span>
                  <div>
                    <div className="export-item-title">CVAT XML 1.1 (Chuẩn nộp bài)</div>
                    <div className="export-item-sub">Đầy đủ 50 điểm đã sửa theo Model AI</div>
                  </div>
                </a>

                <a
                  href={getJsonDownloadUrl(selectedSessionId)}
                  download
                  className="export-dropdown-item"
                  onClick={() => setShowExportMenu(false)}
                >
                  <span className="export-item-icon">📋</span>
                  <div>
                    <div className="export-item-title">JSON Annotations</div>
                    <div className="export-item-sub">File JSON sạch tọa độ chuẩn hóa</div>
                  </div>
                </a>

                <div className="export-dropdown-divider" />

                <a
                  href={getCsvDownloadUrl(selectedSessionId)}
                  download
                  className="export-dropdown-item"
                  onClick={() => setShowExportMenu(false)}
                >
                  <span className="export-item-icon">📊</span>
                  <div>
                    <div className="export-item-title">Báo Cáo QA (CSV)</div>
                    <div className="export-item-sub">Bảng tổng hợp NME và vi phạm</div>
                  </div>
                </a>

                <a
                  href={getChecklistDownloadUrl(selectedSessionId)}
                  download
                  className="export-dropdown-item"
                  onClick={() => setShowExportMenu(false)}
                >
                  <span className="export-item-icon">📝</span>
                  <div>
                    <div className="export-item-title">Checklist Sửa Lỗi (Markdown)</div>
                    <div className="export-item-sub">Hướng dẫn chi tiết từng khung hình</div>
                  </div>
                </a>
              </div>
            )}
          </div>
        )}

        <button className="btn" onClick={onOpenUpload} title="Tải file ZIP và XML/JSON nhãn mới lên">
          📤 Tải Lên
        </button>

        <div
          title={backendHealthy ? 'Backend kết nối tốt (Port 8001)' : 'Backend mất kết nối'}
          style={{
            width: '9px',
            height: '9px',
            borderRadius: '50%',
            backgroundColor: backendHealthy ? 'var(--success)' : 'var(--danger)',
            boxShadow: backendHealthy ? '0 0 6px var(--success)' : '0 0 6px var(--danger)',
            marginLeft: '4px',
          }}
        />
      </div>
    </header>
  );
};
