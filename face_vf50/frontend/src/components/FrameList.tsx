import React from 'react';
import type { FrameListItem } from '../types';

interface FrameListProps {
  frames: FrameListItem[];
  totalFrames: number;
  selectedFrameIndex: number | null;
  searchImage: string;
  onSearchChange: (val: string) => void;
  onSelectFrame: (index: number) => void;
  onStatusChange: (frameIndex: number, newStatus: 'pending' | 'reviewed' | 'fixed') => void;
}

export const FrameList: React.FC<FrameListProps> = ({
  frames,
  totalFrames,
  selectedFrameIndex,
  searchImage,
  onSearchChange,
  onSelectFrame,
  onStatusChange,
}) => {
  return (
    <div className="panel-left">
      {/* Sticky Header with Search and Counters */}
      <div className="frame-list-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--text-primary)' }}>
            Danh Sách Khung Hình ({frames.length}/{totalFrames})
          </span>
        </div>

        <input
          type="text"
          className="frame-search-input"
          placeholder="🔍 Tìm tên file..."
          value={searchImage}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>

      {/* Frame Items List */}
      <div className="frame-items-container">
        {frames.length === 0 ? (
          <div style={{ padding: '28px 14px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            Không tìm thấy khung hình nào khớp với bộ lọc.
          </div>
        ) : (
          frames.map((f) => {
            const isActive = selectedFrameIndex === f.frame_index;
            const hasErrors = f.error_count > 0;
            const isHighNme = f.nme > 0.035;

            return (
              <div
                key={`frame_card_${f.frame_index}`}
                className={`frame-card ${isActive ? 'active' : ''}`}
                onClick={() => onSelectFrame(f.frame_index)}
              >
                <div className="frame-card-title">
                  <span style={{ color: isActive ? 'var(--cyan)' : 'var(--text-primary)' }}>
                    #{f.frame_index} &middot; {f.image_filename}
                  </span>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.76rem',
                      fontWeight: 700,
                      color: isHighNme ? 'var(--danger)' : f.nme > 0.02 ? 'var(--warning)' : 'var(--cyan)',
                    }}
                  >
                    {f.nme.toFixed(3)}
                  </span>
                </div>

                <div className="frame-card-meta">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
                    {hasErrors ? (
                      <span className="badge badge-error" style={{ fontSize: '0.65rem' }}>
                        {f.error_count} vi phạm
                      </span>
                    ) : (
                      <span className="badge badge-success" style={{ fontSize: '0.65rem' }}>
                        Đạt chuẩn
                      </span>
                    )}
                    {f.case_type && f.case_type !== 'normal' && (
                      <span
                        className="badge badge-warning"
                        style={{ fontSize: '0.63rem', padding: '1px 5px' }}
                        title={f.case_label || 'Ca khó / Góc khuất'}
                      >
                        {f.case_type === 'sunglasses' ? '🕶️ Kính' : f.case_type === 'far' ? '🔍 Xa' : '⚠️ Ca khó'}
                      </span>
                    )}
                  </div>

                  {/* Status selector */}
                  <select
                    value={f.status}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) =>
                      onStatusChange(f.frame_index, e.target.value as 'pending' | 'reviewed' | 'fixed')
                    }
                    style={{
                      padding: '1px 5px',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                      borderRadius: '3px',
                      border: '1px solid var(--border-strong)',
                      background:
                        f.status === 'fixed'
                          ? 'var(--success-bg)'
                          : f.status === 'reviewed'
                          ? 'var(--primary-light)'
                          : 'var(--bg-card)',
                      color:
                        f.status === 'fixed'
                          ? 'var(--success)'
                          : f.status === 'reviewed'
                          ? 'var(--cyan)'
                          : 'var(--text-secondary)',
                    }}
                  >
                    <option value="pending">Chưa sửa</option>
                    <option value="reviewed">Đã rà soát</option>
                    <option value="fixed">Đã sửa</option>
                  </select>
                </div>

                {/* Violated Rule Badges */}
                {f.rule_violations && f.rule_violations.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px', marginTop: '2px' }}>
                    {f.rule_violations.slice(0, 3).map((v, vIdx) => (
                      <span
                        key={`rule_badge_${f.frame_index}_${vIdx}`}
                        className="badge badge-warning"
                        style={{ fontSize: '0.64rem', padding: '1px 4px' }}
                      >
                        {v.code}
                      </span>
                    ))}
                    {f.rule_violations.length > 3 && (
                      <span
                        className="badge"
                        style={{ background: 'var(--border-strong)', color: 'var(--text-muted)', fontSize: '0.64rem' }}
                      >
                        +{f.rule_violations.length - 3}
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
