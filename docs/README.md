# Tài liệu dự án Landmark QA

- [`ARCHITECTURE.md`](./ARCHITECTURE.md): cấu trúc sau refactor, ranh giới module, migration, kiểm tra và các nhánh thực hiện.

Nếu đưa dự án này cho một AI khác, hãy cung cấp trước file:

- [`AI_PROJECT_GUIDE.md`](./AI_PROJECT_GUIDE.md): tài liệu nguồn chính, mô tả kiến trúc, dữ liệu, API, database, cách triển khai và toàn bộ công thức Pose/Face theo source hiện tại.

Tài liệu cũ trong `backend/tailieu/` vẫn có giá trị tham khảo nghiệp vụ, nhưng không phải lúc nào cũng khớp mã nguồn. Đặc biệt, bảng ánh xạ R01–R14 trong tài liệu cũ khác implementation hiện tại. Khi có xung đột, ưu tiên theo thứ tự:

1. Test đang chạy.
2. Source trong `backend/app/` và `frontend/src/`.
3. `backend/configs/face_landmark_vf50/vf50_spec.yaml` đối với ngưỡng Face.
4. Tài liệu trong thư mục này.
5. Tài liệu lịch sử ở `backend/tailieu/`, `PROJECT_CONTEXT.md` và `FACE_LANDMARK_VF50_CONTEXT.md`.
