export const renderBadge = (type: string) => {
    if (type === 'swap_error') return <span className="badge badge-swap">Hoán đổi Trái-Phải</span>
    if (type === 'hard_case') return <span className="badge badge-hard">Điểm khó (bị che)</span>
    return <span className="badge badge-error">Nghi sai</span>
  }
