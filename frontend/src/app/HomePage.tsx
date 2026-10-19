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
          <h1 id="home-title" className="hero-title">
            Kiểm định annotation,<span className="hero-title-accent"> tập trung vào quyết định.</span>
          </h1>

          <p className="home-lede">
            Công cụ kiểm chất lượng nhãn Human Pose và Face Landmark: rà soát sai lệch keypoint bằng model AI,
            đối chiếu với nhãn người gán và xuất dữ liệu đã duyệt.
          </p>

          {/* Decorative — landmark wireframe, aria-hidden */}
          <div className="hero-visuals" aria-hidden="true">
          <svg className="hero-visual hero-visual--pose" viewBox="0 0 420 470" fill="none">
            <g stroke="#18181b" strokeOpacity="0.1" strokeWidth="2">
              <circle cx="210" cy="48" r="26" />
              <path d="M210 76v26" />
              <path d="M210 102 122 124" />
              <path d="M210 102 298 124" />
              <path d="M120 124h180" />
              <path d="M120 124 96 192" />
              <path d="M300 124 324 192" />
              <path d="M96 192 110 262" />
              <path d="M324 192 310 262" />
              <path d="M210 102 204 242" />
              <path d="M204 242 150 252" />
              <path d="M204 242 262 252" />
              <path d="M120 124 150 252" />
              <path d="M300 124 262 252" />
              <path d="M150 252 136 352" />
              <path d="M262 252 280 352" />
              <path d="M136 352 146 442" />
              <path d="M280 352 270 442" />
            </g>
            <g fill="#18181b" fillOpacity="0.13">
              <circle cx="120" cy="124" r="4.5" />
              <circle cx="96" cy="192" r="4.5" />
              <circle cx="324" cy="192" r="4.5" />
              <circle cx="310" cy="262" r="4.5" />
              <circle cx="150" cy="252" r="4.5" />
              <circle cx="262" cy="252" r="4.5" />
              <circle cx="136" cy="352" r="4.5" />
              <circle cx="280" cy="352" r="4.5" />
              <circle cx="146" cy="442" r="4.5" />
              <circle cx="270" cy="442" r="4.5" />
              <circle cx="210" cy="48" r="4.5" />
              <circle cx="210" cy="102" r="4.5" />
            </g>
            <circle cx="300" cy="124" r="6" fill="#ea580c" fillOpacity="0.5" />
            <circle cx="300" cy="124" r="10.5" stroke="#ea580c" strokeOpacity="0.3" strokeWidth="2" fill="none" />
            <circle cx="110" cy="262" r="6" fill="#059669" fillOpacity="0.5" />
            <circle cx="110" cy="262" r="10.5" stroke="#059669" strokeOpacity="0.3" strokeWidth="2" fill="none" />
          </svg>
          <svg className="hero-visual hero-visual--face" viewBox="0 0 320 340" fill="none">
            <g stroke="#18181b" strokeOpacity="0.1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M160 34c-55 0-91 41-91 101v33c0 60 37 115 91 139 54-24 91-79 91-139v-33c0-60-36-101-91-101Z" />
              <path d="M69 155c-15-15-20 14-4 37m186-37c15-15 20 14 4 37" />
              <path d="M84 119c19-16 44-18 67-5m18 0c23-13 48-11 67 5" />
              <path d="M88 146c15-13 39-15 55 0-16 11-40 11-55 0Zm89 0c16-15 40-13 55 0-15 11-39 11-55 0Z" />
              <path d="M160 123v55l-13 20c7 8 19 8 26 0l-13-20" />
              <path d="M113 218c16-11 33-12 47-6 14-6 31-5 47 6-11 13-28 20-47 20s-36-7-47-20Z" />
              <path d="M135 220c15-5 35-5 50 0-15 8-35 8-50 0Z" />
              <path d="M160 66v35m-72 45h-18m162 0h18m-90 92v45" strokeDasharray="3 8" />
            </g>
            <g fill="#18181b" fillOpacity="0.16">
              {[
                [84, 119], [100, 110], [118, 106], [136, 108], [151, 114],
                [169, 114], [184, 108], [202, 106], [220, 110], [236, 119],
                [160, 123], [160, 142], [160, 161], [160, 178],
                [88, 146], [100, 139], [116, 136], [132, 139], [143, 146], [132, 152], [116, 155], [100, 152],
                [177, 146], [188, 139], [204, 136], [220, 139], [232, 146], [220, 152], [204, 155], [188, 152],
                [113, 218], [125, 211], [139, 207], [151, 210], [160, 212], [169, 210], [181, 207], [195, 211], [207, 218], [193, 230], [160, 238], [127, 230],
                [135, 220], [145, 217], [155, 219], [165, 219], [175, 217], [185, 220], [175, 224], [145, 224],
              ].map(([cx, cy], index) => <circle key={index} cx={cx} cy={cy} r="3" />)}
            </g>
            <circle cx="204" cy="136" r="5.5" fill="#ea580c" fillOpacity="0.5" />
            <circle cx="204" cy="136" r="10" stroke="#ea580c" strokeOpacity="0.3" strokeWidth="2" />
            <circle cx="139" cy="207" r="5.5" fill="#059669" fillOpacity="0.5" />
            <circle cx="139" cy="207" r="10" stroke="#059669" strokeOpacity="0.3" strokeWidth="2" />
          </svg>
          </div>

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

        <footer className="home-footer">
          <span>LANDMARK QA — DATA QUALITY CONSOLE</span>
          <span>v2.4 · HUMANPOSE-17 · FACE-VF50</span>
        </footer>
      </div>
    </main>
  )
}

