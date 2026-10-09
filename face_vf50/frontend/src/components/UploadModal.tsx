import React, { useState } from 'react';
import { uploadDataset } from '../api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (sessionId: string) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [annotationFile, setAnnotationFile] = useState<File | null>(null);
  const [sessionName, setSessionName] = useState<string>('');
  const [runModel, setRunModel] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!zipFile) {
      setErrorMsg('Vui lòng chọn file ZIP chứa ảnh khung hình.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append('dataset_zip', zipFile);
      if (annotationFile) {
        formData.append('annotation_file', annotationFile);
      }
      if (sessionName.trim()) {
        formData.append('session_name', sessionName.trim());
      }
      formData.append('run_model', runModel.toString());

      const res = await uploadDataset(formData);
      setIsLoading(false);
      onSuccess(res.session_id);
      onClose();
    } catch (err: any) {
      setIsLoading(false);
      setErrorMsg(err.message || 'Lỗi tải lên và xử lý dataset.');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 className="modal-title">Tải Lên Dataset Kiểm Định</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: '20px',
              cursor: 'pointer',
            }}
          >
            &times;
          </button>
        </div>

        <p className="modal-desc">
          Tải lên bộ ảnh (ZIP) và file nhãn xuất từ CVAT (XML hoặc JSON) để hệ thống tự động kiểm tra 14 quy tắc và tính độ lệch chuẩn NME.
        </p>

        {errorMsg && (
          <div
            style={{
              padding: '10px',
              borderRadius: '6px',
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#FB7185',
              fontSize: '12px',
            }}
          >
            {errorMsg}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
              Tên đợt kiểm định (Tùy chọn)
            </label>
            <input
              type="text"
              placeholder="VD: Batch_Face_Week2_Task1"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '8px 12px',
                color: 'white',
                fontSize: '13px',
                outline: 'none',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
              1. File ZIP dataset (.zip) *
            </label>
            <input
              type="file"
              accept=".zip"
              onChange={(e) => setZipFile(e.target.files?.[0] || null)}
              style={{ fontSize: '12px', color: 'var(--text-secondary)' }}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
              💡 Có thể tải trực tiếp file ZIP xuất từ CVAT (hệ thống sẽ tự nhận diện cả ảnh và file nhãn <code>annotations.xml</code> bên trong).
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
              2. File nhãn CVAT XML hoặc JSON (.xml, .json) (Tùy chọn)
            </label>
            <input
              type="file"
              accept=".xml,.json"
              onChange={(e) => setAnnotationFile(e.target.files?.[0] || null)}
              style={{ fontSize: '12px', color: 'var(--text-secondary)' }}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Chỉ cần chọn nếu file nhãn nằm riêng biệt bên ngoài file ZIP.
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
            <input
              type="checkbox"
              id="runModelCheckbox"
              checked={runModel}
              onChange={(e) => setRunModel(e.target.checked)}
            />
            <label htmlFor="runModelCheckbox" style={{ fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
              Chạy MediaPipe Face Landmarker để đối chiếu và tính chỉ số sai số NME
            </label>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' }}>
            <button type="button" className="btn btn-outline" onClick={onClose} disabled={isLoading}>
              Hủy
            </button>
            <button type="submit" className="btn btn-primary" disabled={isLoading}>
              {isLoading ? 'Đang phân tích...' : 'Bắt đầu kiểm định'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
