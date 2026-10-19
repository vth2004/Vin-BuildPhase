import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'

import FaceLandmarkPage from '../features/face-landmark-vf50/FaceLandmarkPage'
import HumanPosePage from '../features/human-pose/HumanPosePage'
import TemporalReviewPage from '../features/temporal/TemporalReviewPage'
import HomePage from './HomePage'

export default function App() {
  const location = useLocation()
  return (
    <div className="app-root">
      <nav className="app-nav" aria-label="Điều hướng chính">
        <div className="nav-container">
          <Link className="brand" to="/" aria-label="Landmark QA trang chủ">
            <span className="brand-mark">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3m0 14v3M2 12h3m14 0h3" />
                <circle cx="12" cy="12" r="8" strokeOpacity="0.4" />
              </svg>
            </span>
            <div className="brand-info">
              <span className="brand-title">LANDMARK QA</span>
              <span className="brand-tag">VINFAST DATA QUALITY</span>
            </div>
          </Link>

          <div className="nav-links">
            <Link className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} to="/">
              Tổng quan
            </Link>
            <Link className={`nav-link ${location.pathname === '/pose' ? 'active' : ''}`} to="/pose">
              <span className="nav-dot nav-dot--pose" aria-hidden="true"></span>
              Human Pose
              <span className="nav-pill">COCO-17</span>
            </Link>
            <Link className={`nav-link nav-link--temporal ${location.pathname === '/pose/temporal' ? 'active' : ''}`} to="/pose/temporal">Pose Temporal</Link>
            <Link className={`nav-link ${location.pathname === '/face' ? 'active' : ''}`} to="/face">
              <span className="nav-dot nav-dot--face" aria-hidden="true"></span>
              Face Landmark
              <span className="nav-pill">VF50</span>
            </Link>
            <Link className={`nav-link nav-link--temporal ${location.pathname === '/face/temporal' ? 'active' : ''}`} to="/face/temporal">Face Temporal</Link>
          </div>

          <div className="nav-meta">
            <span className="status-badge">
              <span className="status-dot" aria-hidden="true"></span>
              v2.4
            </span>
          </div>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/pose" element={<HumanPosePage />} />
        <Route path="/pose/temporal" element={<TemporalReviewPage feature="pose" />} />
        <Route path="/face" element={<FaceLandmarkPage />} />
        <Route path="/face/temporal" element={<TemporalReviewPage feature="face" />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}
