import { Link } from 'react-router-dom'

const tools = [
  {
    code: '01',
    title: 'Human Pose Quality',
    eyebrow: 'COCO · 17 ĐIỂM KEYPOINT',
    tag: 'Ensemble K=2 (YOLO26s + RTMPose)',
    description: 'Kiểm tra sai lệch keypoint toàn thân, đối chiếu nhãn gán người với đề xuất từ mô hình AI, phát hiện đảo chi trái-phải và tính toán dung sai OKS chuẩn hóa.',
    to: '/pose',
    action: 'Mở Human Pose Studio',
    tone: 'pose',
    features: ['17 Khớp toàn thân COCO', 'Phát hiện hoán đổi Trái/Phải', 'Dung sai OKS (σ) động', 'Phím tắt Review K / M / S'],
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="4" r="2" />
        <path d="M12 6v7" />
        <path d="m9 9 3 2 3-2" />
        <path d="m9 17 3-4 3 4" />
        <path d="M9 21v-4m6 0v4" />
      </svg>
    ),
  },
  {
    code: '02',
    title: 'Face Landmark VF50',
    eyebrow: 'VF50 · 50 ĐIỂM NGŨ QUAN',
    tag: 'MediaPipe + 14 Geometric Rules',
    description: 'Rà soát 50 điểm mốc khuôn mặt, tính toán khoảng cách gian tử IOD, sai số chuẩn hóa NME và kích hoạt 14 luật hình học nghiêm ngặt để phát hiện điểm dị biệt.',
    to: '/face',
    action: 'Mở Face Landmark Studio',
    tone: 'face',
    features: ['50 Landmark 7 vùng ngũ quan', '14 Luật hình học tự động', 'Đo khoảng cách IOD & sai số NME', 'Batch Fix AI 1-Click'],
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M8 10h.01M16 10h.01" />
        <path d="M12 11v3" />
        <path d="M9 16c1 1 2 1.5 3 1.5s2-.5 3-1.5" />
      </svg>
    ),
  },
]

const steps = [
  { step: '01', title: 'Import Dataset', desc: 'Hỗ trợ ZIP từ CVAT, file JSON COCO hoặc XML' },
  { step: '02', title: 'Quét Tự Động', desc: 'Chạy Dual-Model Ensemble & 14 quy tắc hình học' },
  { step: '03', title: 'Đối Soát Trực Quan', desc: 'So sánh Ground-Truth vs AI với phím tắt tiện lợi' },
  { step: '04', title: 'Xuất Dữ Liệu Đã Duyệt', desc: 'Bảo toàn nhãn gốc, xuất annotation sạch hoàn chỉnh' },
]

export default function HomePage() {
  return (
    <main className="home-page">
      <div className="home-container">
        {/* Hero Section */}
        <section className="home-hero" aria-labelledby="home-title">
          <div className="hero-pill">
            <span className="hero-pill-badge">VINFAST AUTOMOTIVE AI</span>
            <span className="hero-pill-divider"></span>
            <span className="hero-pill-text">Data Quality Engineering Suite</span>
          </div>

          <h1 id="home-title" className="hero-title">
            Kiểm định annotation,
            <span className="hero-title-accent"> tập trung vào quyết định.</span>
          </h1>

          <p className="home-lede">
            Nền tảng kiểm định chất lượng dữ liệu thị giác máy tính chuyên sâu cho các bộ dữ liệu Human Pose và Face Landmark.
            Tự động rà soát sai lệch keypoint, đối chiếu đề xuất AI và tối ưu hóa thời gian QA cho đội ngũ dữ liệu.
          </p>

          <div className="hero-stats">
            <div className="stat-card">
              <span className="stat-value">COCO-17 & VF50</span>
              <span className="stat-label">Hỗ trợ 2 chuẩn công nghiệp</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">14+ Rules</span>
              <span className="stat-label">Luật kiểm tra hình học tự động</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">OKS & NME</span>
              <span className="stat-label">Chuẩn đo lường độ lệch quốc tế</span>
            </div>
          </div>
        </section>

        {/* Tool Selection Cards */}
        <section className="tool-grid" aria-label="Chọn công cụ kiểm định">
          {tools.map(tool => (
            <article className={`tool-card tool-card--${tool.tone}`} key={tool.to}>
              <div className="tool-card-glow" aria-hidden="true"></div>
              
              <div className="tool-card-header">
                <div className="tool-icon-wrap">
                  {tool.icon}
                </div>
                <div className="tool-meta">
                  <span className="tool-eyebrow">{tool.eyebrow}</span>
                  <span className="tool-tag">{tool.tag}</span>
                </div>
                <span className="tool-index">{tool.code}</span>
              </div>

              <div className="tool-card-body">
                <h2 className="tool-title">{tool.title}</h2>
                <p className="tool-desc">{tool.description}</p>

                <div className="tool-feature-list">
                  {tool.features.map(f => (
                    <span className="tool-feature-chip" key={f}>
                      <span className="chip-check">✓</span> {f}
                    </span>
                  ))}
                </div>
              </div>

              <div className="tool-card-footer">
                <Link to={tool.to} className="tool-button">
                  <span>{tool.action}</span>
                  <span className="tool-button-arrow" aria-hidden="true">→</span>
                </Link>
              </div>
            </article>
          ))}
        </section>

        {/* Workflow Pipeline */}
        <section className="workflow-section" aria-label="Quy trình kiểm định 4 bước">
          <div className="workflow-header">
            <span className="workflow-badge">QUY TRÌNH CHUẨN</span>
            <h3 className="workflow-title">Quy trình kiểm soát chất lượng 4 bước</h3>
          </div>

          <div className="workflow-steps">
            {steps.map((s, idx) => (
              <div className="workflow-step-card" key={s.step}>
                <div className="step-number">{s.step}</div>
                <div className="step-content">
                  <h4 className="step-title">{s.title}</h4>
                  <p className="step-desc">{s.desc}</p>
                </div>
                {idx < steps.length - 1 && <div className="step-connector" aria-hidden="true"></div>}
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}

