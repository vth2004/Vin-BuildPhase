import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'

import FaceLandmarkPage from '../features/face-landmark-vf50/FaceLandmarkPage'
import HumanPosePage from '../features/human-pose/HumanPosePage'

export default function App() {
  const location = useLocation()
  return (
    <>
      <nav className="app-nav" aria-label="Chọn công cụ kiểm định">
        <Link className={location.pathname === '/' ? 'active' : ''} to="/">HumanPose</Link>
        <Link className={location.pathname.startsWith('/face') ? 'active' : ''} to="/face">Face Landmark VF50</Link>
      </nav>
      <Routes>
        <Route path="/" element={<HumanPosePage />} />
        <Route path="/face" element={<FaceLandmarkPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
