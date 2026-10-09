# Quy Tắc Thiết Kế UI & Kiểm Định Trực Quan (Impeccable UI Design Rule)

Quy tắc này **BẮT BUỘC ÁP DỤNG** cho Agent mỗi khi tạo mới, chỉnh sửa hoặc tái cấu trúc giao diện người dùng (Frontend / UI / Web App) trong dự án.

---

## 1. Nguyên Tắc Bắt Buộc (Mandatory Workflow)
Trước khi xuất bản code hoặc đẩy kết quả giao diện vào **Artifacts**:
1. **Luôn chạy Dev Server & Render trên Trình duyệt:**
   - Đảm bảo Frontend dev server đang chạy (ví dụ `http://localhost:5174` hoặc `http://localhost:5173`).
   - Sử dụng công cụ **Browser (`browser_subagent`)** để mở URL, render giao diện thực tế và chụp ảnh màn hình / kiểm tra DOM.
2. **Đối chiếu trực quan với Checklist 24 Lỗi Thiết Kế Phổ Biến (Impeccable Anti-Patterns):**
   - Từng màn hình, từng component chính phải được soi kỹ theo 24 tiêu chí bên dưới nhằm triệt tiêu hoàn toàn "AI slop" (phong cách AI rập khuôn, sến sẩm).
3. **Chỉ hoàn tất / đẩy vào Artifacts khi đã sạch lỗi:**
   - Khắc phục triệt để các lỗi visual hierarchy, spacing, contrast và layout phát hiện qua browser trước khi báo cáo hoàn thành.

---

## 2. Danh Mục 24 Lỗi Thiết Kế Phổ Biến (Impeccable Anti-Patterns)

### Nhóm 1: Typography & Phông Chữ
1. **Generic / Overused Fonts:** Dùng font mặc định hoặc lạm dụng Inter/Arial ở khắp mọi nơi bất kể ngữ cảnh sản phẩm. Cần chọn font có cá tính và phù hợp thương hiệu.
2. **Flat Type Hierarchy:** Phân cấp cỡ chữ và độ đậm mờ nhạt (khoảng cách size giữa H1, H2, H3, Body quá gần nhau khiến trang bị phẳng lì).
3. **Excessive Line Length:** Dòng văn bản quá dài (>75ch) hoặc quá hẹp (<45ch), gây mệt mỏi khi đọc nội dung.
4. **Gradient Text Overuse:** Lạm dụng chữ đổ màu gradient trên các thẻ heading, gây sến và giảm khả năng đọc lướt.

### Nhóm 2: Màu Sắc & Độ Tương Phản (Colors & Contrast)
5. **AI Purple/Violet Gradient Syndrome:** Lạm dụng dải gradient tím - xanh neon (purple-blue) rập khuôn phong cách AI SaaS rẻ tiền.
6. **Pure Black & Pure Gray:** Dùng màu đen thuần (`#000000`) hoặc xám trơ không pha tint. Luôn sử dụng gam xám có sắc thái (slate, zinc, cool-gray tinted với màu chủ đạo).
7. **Gray Text on Colored Backgrounds:** Đặt chữ màu xám trên nền có màu (làm chữ bị đục và bẩn). Chữ phụ trên nền màu phải được tint từ chính màu nền hoặc màu chữ chính.
8. **Contrast Violations (WCAG Failure):** Chữ không đạt độ tương phản tối thiểu (Body < 4.5:1, Title < 3:1). Đặc biệt lưu ý placeholder, caption, metadata và badge.
9. **Zero-offset Glow Halos:** Đổ bóng phát sáng tỏa đều không có offset trục Y/X trông như vầng hào quang lòe loẹt. Bóng đổ phải có offset nhẹ và độ mờ tự nhiên (soft ambient elevation).

### Nhóm 3: Bố Cục & Cấu Trúc Trang (Layout & Scaffolds)
10. **Nested Cards (Thẻ lồng thẻ):** Nhét card bên trong card nhiều tầng gây rối mắt, tạo cảm giác ngột ngạt và lãng phí không gian.
11. **Identical Card Grids:** Nhồi nhét mọi nội dung thành các ô thẻ vuông vức giống hệt nhau (icon + title + text), thiếu nhịp điệu và tiêu điểm thị giác.
12. **Everything-Centered Layout:** Căn giữa toàn bộ nội dung một cách máy móc, làm mất luồng đọc tự nhiên F-shape / Z-shape của mắt.
13. **Hero-metric Template Syndrome:** Lạm dụng khuôn mẫu "Số to đùng + Nhãn nhỏ xíu + Icon tròn" một cách vô hồn.
14. **Side-stripe Accent Borders:** Dùng dải màu nhấn ở mép trái của card/thẻ (side-tab borders) kiểu UI cũ kỹ, thiếu tinh tế.

### Nhóm 4: Khoảng Cách, Căn Chỉnh & Bo Góc (Spacing & Rhythm)
15. **Monotonous / Inconsistent Spacing:** Khoảng cách giữa các phần tử đều nhau chằn chặn, vi phạm luật gần gũi (Gestalt - khoảng cách cùng nhóm phải nhỏ hơn khoảng cách giữa các khối khác nhau).
16. **Cramped Padding:** Padding bên trong card/button quá chật chội, chữ và icon dính sát mép viền ngoài.
17. **Heading Asymmetry Ignored:** Khoảng cách phía trên heading không lớn hơn khoảng cách bên dưới heading (khiến heading bị "dính" vào nội dung của khối trước).
18. **Clashing Border-Radius:** Sử dụng bán kính bo góc lộn xộn (ví dụ: nút bấm bo pill 9999px nhưng card bo 4px nhọn hoắt; hoặc card cha bo góc nhỏ hơn card con bên trong).

### Nhóm 5: Thành Phần & Hoàn Thiện (Components & Polish)
19. **Rounded-Square Icon Tiles:** Đặt các icon nằm lọt thỏm trong khối vuông bo góc có nền nhạt phía trên mỗi heading một cách rập khuôn.
20. **Unstyled Browser Surfaces:** Bỏ quên các bề mặt mặc định của trình duyệt: thanh cuộn (scrollbar) thô kệch, vùng bôi đen (`::selection`), caret màu, focus outline mặc định lòe xòe. Phải style đồng bộ với design system.
21. **Glassmorphism as Default:** Lạm dụng hiệu ứng kính mờ (backdrop-blur + viền trắng bán trong suốt) tràn lan khắp nơi ngay cả khi nền không có texture.

### Nhóm 6: Chuyển Động, Trạng Thái & Nội Dung (Motion, States & Copy)
22. **Dated Bounce / Elastic Easing:** Dùng animation nảy tưng tưng giật cục. Chỉ sử dụng chuyển động mượt mà (smooth cubic-bezier, ease-out tinh tế).
23. **Scattered / Inconsistent Animations:** Mỗi thành phần xuất hiện một kiểu lộn xộn (vừa xoay, vừa nảy, vừa fade). Chuyển động phải đồng nhất và có chủ đích rõ ràng.
24. **Missing UI States & Robotic Copy:** Bỏ quên các trạng thái thiết yếu: Hover, Active, Focus-visible, Loading/Skeleton, Empty state, Error state; hoặc câu thông báo lỗi cứng nhắc, thiếu hướng dẫn khắc phục.

---

## 3. Checklist Thực Thi Nhanh Khi Kiểm Tra Trình Duyệt
- [ ] Đã mở trình duyệt thực tế và render đúng responsive breakpoint (Desktop / Laptop)?
- [ ] Đã soi kỹ độ tương phản text và màu nền (không có chữ xám trên nền màu)?
- [ ] Không có card lồng card, không có grid card đơn điệu?
- [ ] Spacing nhịp nhàng, heading gắn chặt với nội dung bên dưới hơn là khối bên trên?
- [ ] Đã kiểm tra đầy đủ hover states, empty states, loading indicators?
- [ ] Scrollbars và selection đã được custom ăn nhập với theme?
