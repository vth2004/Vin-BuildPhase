# Workspace Guidelines & Agent Rules

## 1. UI Design & Visual Verification Rule (Bắt buộc khi làm Frontend/UI)
Khi phát triển, chỉnh sửa hoặc tạo giao diện người dùng (UI):
- **BẮT BUỘC** luôn khởi động dev server và dùng công cụ **Browser** (`browser_subagent`) để render UI trực quan trên trình duyệt (ví dụ: `http://localhost:5174` hoặc `http://localhost:5173`).
- **BẮT BUỘC** kiểm tra đối chiếu trực quan xem giao diện có mắc phải **24 lỗi thiết kế phổ biến theo bộ quy tắc Impeccable Skill** (chi tiết xem tại [.agents/rules/ui_design_impeccable.md](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/.agents/rules/ui_design_impeccable.md)):
  1. *Generic / Overused Fonts* (tránh lạm dụng Inter/Arial rập khuôn).
  2. *Flat Type Hierarchy* (phân cấp cỡ chữ, độ đậm rõ rệt).
  3. *Excessive Line Length* (độ dài dòng chuẩn 45–75 ký tự).
  4. *Gradient Text Overuse* (không lạm dụng chữ gradient trên heading).
  5. *AI Purple/Violet Gradient Syndrome* (không dùng dải gradient tím - xanh neon AI rẻ tiền).
  6. *Pure Black & Pure Gray* (dùng tinted neutrals thay cho #000000 và xám trơ).
  7. *Gray Text on Colored Backgrounds* (chữ phụ phải được tint từ nền/chữ chính, không dùng xám xỉn).
  8. *Contrast Violations* (đảm bảo chuẩn WCAG AA: body ≥ 4.5:1, large text ≥ 3:1).
  9. *Zero-offset Glow Halos* (bóng đổ có offset Y/X và blur mềm mại, tránh quầng hào quang phát sáng).
  10. *Nested Cards* (tuyệt đối không dùng thẻ lồng trong thẻ).
  11. *Identical Card Grids* (tránh lưới card icon+title+text đơn điệu, tạo nhịp điệu bố cục đa dạng).
  12. *Everything-Centered Layout* (bố cục theo dòng đọc tự nhiên F/Z-pattern).
  13. *Hero-metric Template Syndrome* (tránh cụm số to nhãn nhỏ rập khuôn).
  14. *Side-stripe Accent Borders* (tránh viền màu nhấn dọc mép thẻ cũ kỹ).
  15. *Monotonous / Inconsistent Spacing* (tuân thủ luật gần gũi Gestalt).
  16. *Cramped Padding* (padding rộng rãi, thoáng đãng).
  17. *Heading Asymmetry Ignored* (khoảng cách trên heading phải lớn hơn khoảng cách dưới heading).
  18. *Clashing Border-Radius* (hài hòa bán kính bo góc giữa các phần tử cha-con).
  19. *Rounded-Square Icon Tiles* (tránh ô icon bo tròn rập khuôn trên mỗi tiêu đề).
  20. *Unstyled Browser Surfaces* (tùy biến scrollbar, selection, focus indicator ăn nhập với theme).
  21. *Glassmorphism as Default* (không lạm dụng kính mờ vô tội vạ).
  22. *Dated Bounce / Elastic Easing* (chuyển động mượt mà, tránh hiệu ứng nảy giật cục).
  23. *Scattered Animations* (chuyển động đồng nhất, có chủ đích).
  24. *Missing UI States & Robotic Copy* (đủ trạng thái hover, active, focus, loading, empty, error; copy tự nhiên).
- **KHÔNG ĐƯỢC** đẩy code vào Artifacts hoặc báo cáo hoàn thành nếu chưa qua bước render và xác thực trực quan bằng Browser.
