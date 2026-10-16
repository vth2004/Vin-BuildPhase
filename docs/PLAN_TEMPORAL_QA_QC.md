# Kế hoạch bổ sung Temporal QA/QC cho Pose và Face (bản rút gọn v3)

- **Nhánh triển khai:** `dev/hiep-temporal-qa`
- **Trạng thái:** Phase 5 hoàn thành ngày 2026-10-16; Q1–Q2 là quyết định tạm thời của nhóm, chờ mentor xác nhận.
- **Phạm vi:** Human Pose 17 điểm và Face Landmark VF50.

## 1. Mục tiêu và phạm vi

Bổ sung Temporal QA/QC để đánh giá keypoint dựa trên chuyển động qua nhiều frame, phát hiện điểm nhảy bất thường, cung cấp vị trí tham khảo khi bị che ngắn hạn và hỗ trợ reviewer chỉnh nhãn.

<table>
  <tr><th>Dữ liệu</th><th>Xử lý</th><th>Đầu ra</th></tr>
  <tr><td>Frame/video có annotation</td><td>QA hiện tại kết hợp bằng chứng temporal</td><td>Cảnh báo lỗi nhãn, vị trí gợi ý, kết quả review</td></tr>
  <tr><td>Frame/video chưa có annotation</td><td>Kiểm tra tính ổn định của dự đoán AI</td><td>Cảnh báo dự đoán bất thường và bộ nhãn AI nháp</td></tr>
</table>

### Các ràng buộc đã chốt

- Temporal QA là tùy chọn khi tạo run, mặc định tắt. Khi tắt, kết quả tương đương baseline hiện tại.
- Phân tích offline hai chiều, dùng frame trước và sau.
- Cửa sổ mặc định **15 frame**, số lẻ **5–29 frame**.
- Reviewer có thể chấp nhận tọa độ temporal theo từng điểm để đưa vào export (Phase 6).
- Chưa có batch fix temporal, realtime hoặc huấn luyện mô hình temporal.
- Tính năng có nhãn **thử nghiệm** cho đến khi có benchmark temporal riêng.
- **Không phụ thuộc track của công cụ gán nhãn.** Nhãn được vẽ dạng shape trong CVAT và export COCO Keypoints 1.0, không có track ID. Việc liên kết cùng một người qua các frame do hệ thống tự làm bằng matching hình học trên dự đoán của model.
- **ZIP-first là UX ưu tiên.** Hệ thống tự suy sequence và người dùng xác nhận; vẫn giữ upload annotation rời hiện có, còn manifest là tùy chọn để ghi đè hoặc xử lý trường hợp mơ hồ.
- Chấp nhận matching không hoàn hảo: khi mơ hồ thì bỏ qua điểm đó, không ép ghép và không tạo cảnh báo.

**Cửa sổ quan sát và thời gian khôi phục che khuất là hai cấu hình khác nhau.** Cửa sổ 29 frame không đồng nghĩa được dự đoán xuyên qua 29 frame bị che.

## 2. Quyết định Phase 0

<table>
  <tr><th>#</th><th>Quyết định ngày 2026-10-16</th><th>Trạng thái/chủ sở hữu</th></tr>
  <tr><td>Q1</td><td>Temporal phục vụ cả khôi phục reference khi model frame hiện tại yếu và phát hiện annotation nhảy; khôi phục reference là mục đích chính. <code>annotation_jitter</code> chỉ là evidence thử nghiệm, không đổi spatial score.</td><td>Nhóm dự án chốt tạm thời; chờ mentor xác nhận</td></tr>
  <tr><td>Q2</td><td>Dùng robust local trajectory fit bậc 2, fallback bậc 1, trong tọa độ chuẩn hóa theo track. Không dùng trung bình thuần.</td><td>Nhóm dự án chốt tạm thời; chờ mentor xác nhận</td></tr>
  <tr><td>Q3</td><td>Demo chính một người/một hành động với fast-path có continuity gate; tracker và test vẫn hỗ trợ nhiều người.</td><td>Đã chốt</td></tr>
  <tr><td>Q4</td><td>CI dùng fixture tổng hợp xác định cho Pose và Face. Smoke test thủ công dùng Penn Action sequence 0001, frame 000001–000030 cho Pose và 300VW sequence 001 + 410, 30 frame đầu cho Face. Không dùng dữ liệu VinFast và không commit dữ liệu research-only.</td><td>Đã chốt; checksum archive ghi khi acquire</td></tr>
  <tr><td>Q5</td><td>COCO thật không có trường frame. Chỉ suy candidate mapping từ filename; COCO array order/numeric ID chỉ là gợi ý. Mapping mơ hồ bắt buộc manifest.</td><td>Đã xác minh trên các ZIP hiện có</td></tr>
  <tr><td>Q6</td><td>Không dùng parser CVAT XML của Face cho Pose. Pose tiếp tục nhận COCO Keypoints 1.0/JSON; chỉ viết adapter XML riêng nếu sau này xuất hiện nhu cầu thật.</td><td>Đã chốt</td></tr>
</table>

### Bằng chứng Q4–Q6

- **CI fixture:** sinh bằng code với seed cố định; gồm chuyển động đứng yên, đều, đổi hướng, outlier, gap, scene cut, một người và hai người giao nhau. Fixture/expected output được commit; không cần model hoặc network.
- **Pose smoke fixture:** [Penn Action](https://github.com/dreamdragon/PennAction/blob/master/README) công bố chuỗi RGB liên tiếp và annotation 2D/visibility/bbox nhưng README nguồn không nêu license minh thị. Chỉ dùng research-only, map 13 body joints có ngữ nghĩa sang VF17 và đặt điểm không ánh xạ là `outside`. Ghi SHA-256 archive, URL và ngày tải trong fixture manifest trước lần chạy đầu.
- **Face smoke fixture:** [300VW](https://www.ibug.doc.ic.ac.uk/resources/300-VW/) dùng research-only, không commit ảnh. Chọn sequence `001` để kiểm tra chuyển động tương đối ổn định và `410` làm ca chuyển động/pose khó; mỗi sequence lấy 30 frame đầu. Archive đã dùng trong benchmark có SHA-256 `8dd5b8a200a4d9804253be79f504100a0305b87510363de7d1b79da6c6d8dbe0`; mapping dùng `experiments/benchmark_face/configs/ibug68_to_vf50.yaml`.
- **COCO thực tế đã kiểm tra:** `humanposenew.zip` có `train_01.jpg`…`train_20.jpg`, numeric image ID 1…20 và không có `frame`/`frame_id`; có thể tạo một candidate sequence nhưng vẫn phải xác nhận. Các ZIP Face và ZIP hai ảnh cũng không có trường frame; tên ảnh/ID không đủ chứng minh quan hệ thời gian và có trường hợp trộn nhiều prefix/ảnh rời.
- **Parser:** Pose hiện hỗ trợ COCO/JSON và chủ động từ chối XML; parser CVAT XML thuộc feature Face. Giữ ranh giới feature, không import parser Face vào Pose.

**Kết luận Phase 0:** hoàn thành ngày 2026-10-16. Q1–Q2 có hiệu lực như quyết định tạm thời để triển khai; nếu mentor thay đổi ý nghĩa hoặc thuật toán, dừng trước Phase 3 để cập nhật kế hoạch/config version.

## 3. Thiết kế

### 3.1. Hiện trạng cần xử lý

- Pose inference/ensemble theo từng ảnh, chưa có person tracking xuyên frame.
- Pose ghép prediction với người đã gán nhãn bằng khoảng cách tâm (`match_person_for_model`), lấy cái gần nhất; không bảo đảm một-một và không có trạng thái mơ hồ. Ensemble có lọc đồng thuận bbox (IoU ≥ 0.30) để loại model bắt nhầm người.
- `p_star` hiện chỉ được tính sau khi từng model đã ghép prediction với annotation. Vì vậy reference hiện tại phụ thuộc annotation và không thể tái sử dụng trực tiếp làm temporal observation. Temporal phải tạo observation stream theo từng detection, độc lập annotation.
- Pose hiện có thể bắt exception rồi sinh mock result. Production run phải chuyển failed; mock chỉ dùng khi test/dev bật rõ ràng.
- Face ghép mặt bằng Hungarian một-một (`_match_faces`, cost là khoảng cách centroid, ngưỡng cắt 2.0 × IOD hoặc 150 px, cấu hình trong `vf50_spec.yaml`), nhưng chỉ trong một frame. Face có rule temporal jump (R13) với frame trước, chưa bảo đảm hai frame là cùng một người và chưa khôi phục tọa độ từ chuỗi.
- Face đã hỗ trợ một phần ảnh không annotation; Phase 8 mở rộng temporal QC, chuẩn hóa AI draft và bổ sung Pose, không xây lại Face từ đầu.
- `frame_index` đang gán theo thứ tự import, không đủ xác nhận sequence video.
- Một số bước chuẩn hóa annotation làm rơi metadata frame/instance.
- Dữ liệu nội bộ sau parse là dict/list chuẩn hóa (không dùng tensor): Face `{point_id: {x, y, state}}`, Pose `[{id, name, x, y, state}]`; scoring không biết model là gì. Temporal làm việc trực tiếp trên dạng này.
- MediaPipe không có confidence đáng tin cậy theo từng landmark; không tự gán 1.0.
- Job chạy trong process web bằng FastAPI background task, không phải durable queue.

### 3.2. Luồng xử lý

```text
ZIP frame (hoặc video, Phase 7) + annotation tùy chọn
                    ↓
  Tự suy sequence/thứ tự frame → thẻ tóm tắt → người dùng xác nhận
                    ↓
         Inference Pose hoặc Face
                    ↓
   Ghép annotation ↔ detection trong từng frame (một-một)
                    ↓
       Nối detection giữa các frame thành track (model-based)
                    ↓
       Ước lượng quỹ đạo trong cửa sổ 5–29
                    ↓
      QA annotation hoặc QC dự đoán AI
                    ↓
       Review → chấp nhận điểm → export


```

### 3.3. Tổ chức code

- Thuật toán chung trong `backend/app/shared/temporal/`: cấu trúc dữ liệu, suy sequence, matching, tracking, time window, trajectory estimation và temporal result.
- Pose và Face giữ adapter, geometry rule và scoring riêng; không import nghiệp vụ của nhau.
- **Matcher mới nằm trong** `shared/temporal/` **và chỉ dùng khi Temporal bật.** Không sửa `match_person_for_model` và `_match_faces` hiện có, để Temporal tắt vẫn cho kết quả tương đương baseline. Nếu muốn dùng chung matcher mới cho luồng cũ thì phải có test và chạy lại benchmark.
- MediaPipe giữ chế độ IMAGE; temporal xử lý bên ngoài adapter.
- Mock inference/result chỉ gọi từ test hoặc development config rõ ràng. Lỗi production phải giữ nguyên nguyên nhân và làm run thất bại.

### 3.4. Metadata và database

Migration có backup, version riêng và chạy lại an toàn:

- **Sequence:** dataset, nguồn ZIP/video, FPS (video, người dùng nhập hoặc không biết), nguồn thứ tự frame, `confirmed_at`, `confirmed_project_id`, `confirmation_source`, snapshot mapping đã xác nhận, trạng thái import và đường dẫn media. Hệ thống hiện không có user identity nên không lưu `confirmed_by_user`.
- **Image:** sequence, frame gốc, timestamp và nguồn metadata.
- **Annotation:** source instance/track ID nếu nguồn có (tùy chọn, không phụ thuộc vào đó), và track nội bộ sau khi ghép.
- **Run:** temporal config và algorithm version, bất biến sau khi tạo.
- **Temporal result:** run, sequence, segment, track, image, annotation, keypoint; evidence/trajectory có thể là JSON, trường cần lọc phải là cột.
- **Temporal review:** run, annotation, keypoint, tọa độ lúc chấp nhận, thời điểm review.
- **Video import job:** dataset, attempt, trạng thái, progress và lỗi; tách khỏi QA run.

Lưu kết quả cho mọi điểm trong phạm vi phân tích, kể cả khi không sinh cảnh báo.

Không dùng `face_index`/thứ tự detection làm track ID. Thêm `frame_index_source`:

<table>
  <tr><th>Giá trị</th><th>Nguồn</th><th>Đủ điều kiện temporal</th></tr>
  <tr><td>manifest</td><td>Manifest người dùng tải lên</td><td>Có, nếu hợp lệ</td></tr>
  <tr><td>video</td><td>Video gốc</td><td>Có, nếu hợp lệ</td></tr>
  <tr><td>annotation</td><td>Trường frame trong file annotation (nếu có)</td><td>Có, nếu hợp lệ</td></tr>
  <tr><td>filename_inferred</td><td>Số trong tên file ảnh</td><td>Chỉ sau khi người dùng xác nhận</td></tr>
  <tr><td>coco_array_order</td><td>Thứ tự phần tử trong <code>images[]</code> của COCO</td><td>Chỉ là gợi ý, sau khi xác nhận</td></tr>
  <tr><td>coco_numeric_id</td><td>Thứ tự khi sort theo image ID số</td><td>Chỉ là gợi ý, sau khi xác nhận</td></tr>
  <tr><td>import_order</td><td>Chỉ là thứ tự import</td><td>Không</td></tr>
</table>

Backfill dữ liệu cũ thành `import_order`. Temporal yêu cầu uniqueness của `(sequence_id, original_frame_index)`. Timestamp lưu integer microseconds; `timestamp_source` là `video`, `manifest`, `fps_derived` hoặc `unavailable`. Dataset cũ vẫn dùng QA hiện tại; muốn bật temporal phải có sequence hợp lệ và đã xác nhận, hoặc import lại.

### 3.5. Ghép annotation, tracking và phân đoạn

**Không dùng track của nguồn.** Source track ID (nếu file có) chỉ là metadata tùy chọn; hệ thống không phụ thuộc vào đó và không dùng nó để tạo temporal reference.

**Bước 1: tạo observation stream độc lập annotation**

- Face có một model nên mỗi face detection là một observation candidate.
- Pose dùng `yolo26s-pose` làm primary: primary detection cung cấp bbox và danh tính track. Detection của model phụ được ghép một-một với primary detection trong cùng frame bằng IoU và cổng hình học, không ghép qua annotation.
- Với mỗi primary detection/keypoint, fuse các model đồng thuận thành `p_star`; đây là temporal observation. Lưu `observation_source` (`primary` hoặc `ensemble`), `supporting_model_count`, danh sách model hỗ trợ và ensemble disagreement.
- Nếu primary detection mất thì ngắt track ở bản đầu, không âm thầm chuyển sang secondary. Chính sách bảo thủ này giảm ID switch nhưng cũng giảm coverage ở frame che nặng; secondary bridge được hoãn sang kế hoạch tương lai.

**Bước 2: ghép annotation ↔ observation trong từng frame**

- Ghép một-một bằng Hungarian. Face tái sử dụng cost centroid và ngưỡng cắt 2.0 × IOD / 150 px của matcher hiện tại, nhưng bổ sung ambiguity margin. Pose dùng cost kết hợp khoảng cách tâm và IoU bbox để giảm nhầm khi hai người giao nhau.
- Có fast-path 1 annotation ↔ 1 detection.
- **Trạng thái mơ hồ:** khi cost tốt nhất và tốt nhì chênh nhau dưới một margin cấu hình, hoặc không vượt ngưỡng cắt, thì không ghép. Annotation đó nhận `tracking_uncertain` và bị bỏ qua trong temporal. Annotation hoặc detection không ghép được vẫn được giữ riêng, không gán bừa.
- Sau khi ghép, annotation thừa hưởng track của detection tương ứng.
- Annotation chỉ dùng để xác định danh tính, không dùng để tạo temporal reference.

**Bước 3: nối primary detection giữa các frame**

- Hungarian một-một giữa primary detection ở các frame liên tiếp, kết hợp bbox overlap, khoảng cách tâm và keypoint similarity. Face dùng face detection; Pose dùng `yolo26s-pose` detection.
- Matching mơ hồ/không đạt ngưỡng thì kết thúc track và tạo track mới.
- Ngắt lịch sử tại sequence mới, scene cut, frame thiếu hoặc đổi kích thước.
- Chưa re-identification sau khi toàn bộ đối tượng biến mất.
- Keypoint bị che vẫn có thể estimate nếu phần còn lại đủ giữ track.
- Scene cut dùng histogram và tỷ lệ matching thất bại; ngưỡng có version và ghi chưa hiệu chỉnh.

**Fast-path một người:** chỉ áp dụng khi mỗi frame có đúng một annotation và đúng một observation/detection đã ghép, detection liên tiếp vượt cổng continuity hình học, không có scene cut và không có gap không hợp lệ. Fast-path vẫn kiểm tra continuity, không nối theo thứ tự thuần túy. Nếu bất kỳ điều kiện nào thất bại thì quay về tracker đầy đủ; nếu vẫn mơ hồ thì tách track hoặc trả `tracking_uncertain`.

R13 Face chỉ so sánh hai frame đã được tracker xác nhận cùng track; không ghép bằng `face_index` hoặc "face gần nhất" độc lập.

### 3.6. Temporal estimation

Dùng robust local trajectory fitting:

1. Thu thập model observation độc lập annotation, cùng keypoint/track trong cửa sổ. Với Pose đây là `p_star` theo primary detection; với Face là landmark của face detection.
2. Dùng thời gian tương đối: timestamp hợp lệ → suy từ FPS → frame index; ghi `time_source`.
3. Chuẩn hóa theo track:

   - Pose: gốc trung điểm hông, scale độ dài thân hông–vai; fallback bbox.
   - Face: gốc tâm bbox, scale IOD; fallback kích thước bbox, cuối cùng image coordinates.
   - Một fit phải dùng cùng `normalization_source`; không trộn IOD/bbox/image coordinates.
4. Hồi quy cục bộ có trọng số với Huber loss (IRLS): bậc 2 khi ≥ 5 observation, bậc 1 khi 2–4; extrapolation chỉ bậc 1.
5. Trọng số giảm theo khoảng cách thời gian và chất lượng observation.
6. Leave-one-out: loại model observation của frame đang kiểm tra khỏi reference.
7. Không dùng annotation đang kiểm tra hoặc review decision để tạo reference.
8. Biến đổi estimate về ảnh bằng origin/scale của frame đích; thiếu transform tin cậy thì trả `unavailable`.

Lưu origin, scale, normalization/time source của frame đích để truy vết.

**Giới hạn cửa sổ:**

- Tối thiểu 5; sau leave-one-out còn tối đa 4 observation. Cửa sổ 3 không đủ để Huber có ý nghĩa.
- Cửa sổ w có (w−1)/2 frame mỗi phía.
- Extrapolation cần ≥ 3 observation cùng phía; cửa sổ < 7 luôn `unavailable` cho extrapolation.
- UI hiển thị thời lượng tương ứng khi biết FPS.

Pose dùng keypoint confidence, ensemble agreement và số model hỗ trợ. Face dùng matching quality, geometry và trajectory residual; ghi nhận confidence thiếu nếu adapter không cung cấp.

**Kết quả mỗi điểm:**

- Raw model coordinate/confidence nếu có.
- `observation_source`, `supporting_model_count` và model provenance.
- Temporal coordinate hoặc `null`.
- `estimate_status`: `available` hoặc `unavailable`.
- `temporal_quality` 0–1, là heuristic không phải xác suất đã hiệu chỉnh.
- Support count, gap, residual.
- Method: `interpolation`, `extrapolation`, `unavailable`.
- `unavailable_reason`: `tracking_uncertain`, `low_support`, `gap_too_long`, `geometry_implausible`, `normalization_unavailable`, `window_too_small`.

Công thức quality, time weight, Huber delta và geometry threshold nằm trong config có version; giá trị ban đầu là mặc định chưa hiệu chỉnh.

**Gap policy mặc định:**

- Interpolation cần observation hợp lệ ở cả hai phía.
- Extrapolation cần ≥ 3 observation và cửa sổ ≥ 7.
- Khi có timestamp hợp lệ, gap tính theo thời gian và đồng thời kiểm tra delta của `original_frame_index`: tối đa 5 original frames và 200 ms; vượt một giới hạn là từ chối.
- Khi không có timestamp, gap tính theo delta của `original_frame_index`, tối đa 3 original frames; không tính theo số ảnh đã import hoặc số frame đã lấy mẫu.
- Nếu người dùng xác nhận dữ liệu được sampling đều và nhập sampled FPS, timestamp suy từ sampled FPS được dùng cho trục thời gian; `original_frame_index` và delta gốc vẫn được giữ để audit.

### 3.7. QA/QC và scoring

- Giữ nguyên coordinate và spatial score gốc.
- Chỉ có một warning mới: `temporal_inconsistency`.
- `evidence`: `model_trajectory`, `annotation_jitter`, `face_r13` hoặc nhiều nguồn.
- `temporal_unavailable` không phải warning; đó là trạng thái kết quả với reason, thống kê trong summary nhưng không vào hàng chờ lỗi nhãn. `tracking_uncertain` do ghép annotation mơ hồ cũng thuộc nhóm này.
- Temporal suggestion là dữ liệu của điểm, không phải cảnh báo và không tự chứng minh annotation sai.

#### `annotation_jitter`

- Xét đạo hàm bậc hai annotation trong tọa độ chuẩn hóa; không dùng annotation tạo model trajectory.
- Cần ≥ 3 annotation hợp lệ, frame/time liên tục, cùng track và normalization source.
- Reset qua gap, scene cut hoặc đổi kích thước.
- So gia tốc điểm với chuyển động cục bộ của điểm lân cận hoặc median vùng cơ thể/khuôn mặt để giảm false positive.
- Ban đầu là evidence thử nghiệm với ngưỡng bảo thủ; không thay đổi spatial score.

Temporal score là khoảng cách chuẩn hóa annotation–temporal coordinate kèm quality. Công thức, warning threshold và xử lý quality thấp được chốt ở Phase 4, lưu theo algorithm/config version.

Với annotation `occluded`, hoặc model frame hiện tại confidence thấp/ensemble bất đồng cao, UI hiển thị temporal score cạnh spatial score. Mặc định vẫn sort spatial; cho chọn sort temporal; chưa cộng hai score tùy ý.

Nếu R13, jitter và model trajectory cùng chỉ ra một keypoint/frame, tạo một `temporal_inconsistency` chứa nhiều evidence, không đếm trùng conclusion. Dữ liệu không annotation chỉ cảnh báo prediction AI; không tính NME hay gọi là lỗi annotation.

### 3.8. Input, API và UI

**Cách người dùng đưa dữ liệu vào:**

1. Chọn chế độ: **Đã có nhãn (COCO Keypoints 1.0)** hoặc **Chưa có nhãn**.
2. Thả ZIP; annotation có thể nằm trong ZIP hoặc tải rời như luồng hiện tại. Video được hỗ trợ từ Phase 7. Người dùng không bắt buộc viết manifest.
3. Hệ thống hiển thị **thẻ tóm tắt**: số sequence, số frame, nguồn thứ tự, frame đầu/cuối, phân bố delta frame, gap đều/không đều và số annotation mỗi frame. Không tự kết luận gap là sampling hay thiếu frame.
4. Hiển thị thumbnail frame đầu/cuối và quanh gap bất thường. Nếu có nhiều numeric token, prefix mơ hồ, trùng frame hoặc không xác định được cách nhóm sequence thì không cho xác nhận tự động và yêu cầu manifest.
5. Người dùng có thể nhập FPS và xác nhận riêng “sampling đều”. Chỉ khi xác nhận sampling đều mới suy timestamp theo sampled FPS.
6. Sau khi xác nhận mapping, toggle Temporal QA khả dụng. Nếu không suy được thứ tự, QA thường vẫn chạy, Temporal bị tắt và UI cho tải manifest.

**Thứ tự ưu tiên khi suy sequence:**

<table>
  <tr><th>Ưu tiên</th><th>Nguồn</th><th>Ghi chú</th></tr>
  <tr><td>1</td><td>Manifest (nếu có)</td><td>Ghi đè mọi nguồn khác</td></tr>
  <tr><td>2</td><td>Trường frame trong annotation (nếu có)</td><td>Đáng tin, không cần xác nhận riêng</td></tr>
  <tr><td>3</td><td>Số trong file_name, sắp xếp tự nhiên, nhóm sequence theo tiền tố</td><td>Cần xác nhận; mơ hồ thì bắt buộc manifest</td></tr>
  <tr><td>4</td><td>Thứ tự phần tử trong COCO images[]</td><td>Chỉ là gợi ý, cần xác nhận</td></tr>
  <tr><td>5</td><td>Sort theo numeric image ID của COCO</td><td>Gợi ý riêng, không đồng nhất với thứ tự images[]</td></tr>
  <tr><td>6</td><td>Không suy ra được</td><td>Temporal tắt</td></tr>
</table>

Video không cần bước suy thứ tự vì thứ tự, FPS và timestamp có sẵn trong video.

**Manifest** (`sequence_manifest.json`, tùy chọn): ánh xạ ảnh → sequence → frame gốc → timestamp tùy chọn. Dùng khi tên file không có số, khi cần ghi đè, hoặc khi đã lấy mẫu frame.

<table>
  <tr><th>Giao diện</th><th>Thay đổi</th><th>Phase</th></tr>
  <tr><td>Upload ZIP</td><td>Tự suy sequence; nhận manifest tùy chọn để ghi đè</td><td>1</td></tr>
  <tr><td>Sequence summary / xác nhận</td><td>Trả thẻ tóm tắt; nhận xác nhận và FPS</td><td>1 (API), 5 (UI)</td></tr>
  <tr><td>Tạo run</td><td>Body nhận temporal_enabled, window_size; lưu config bất biến</td><td>4</td></tr>
  <tr><td>Temporal results</td><td>Phân trang, lọc sequence/track/frame/keypoint/warning/status</td><td>4</td></tr>
  <tr><td>Temporal detail</td><td>Trả trajectory, transform và frame lân cận</td><td>4</td></tr>
  <tr><td>Temporal summary</td><td>Coverage, unavailable reason, warning</td><td>4</td></tr>
  <tr><td>Review điểm</td><td>Chọn/đổi/bỏ temporal</td><td>6</td></tr>
  <tr><td>Upload video</td><td>Tạo dataset/import job và theo dõi progress</td><td>7</td></tr>
</table>

Endpoint mới dùng recovery key và project authorization hiện tại.

**Trên Pose và Face:**

- Upload ZIP, annotation tùy chọn, thẻ tóm tắt sequence.
- Toggle Temporal QA có nhãn thử nghiệm; cửa sổ lẻ 5–29; hiển thị thời lượng và lý do metadata không đủ.
- Overlay Human/Model/Temporal, trajectory có chiều thời gian, timeline thumbnail, frame trước/sau.
- Hiển thị track, quality, normalization/time source và reason.
- Tải frame lân cận theo nhu cầu; giữ frame hiện tại đúng vị trí timeline.
- Summary theo sequence: coverage, warning, gap, tracking uncertainty, unavailable reason và review decision.
- Dùng điểm Temporal chỉ bật khi estimate hợp lệ (Phase 6).

### 3.9. Export (Phase 6)

- Chỉ áp dụng temporal coordinate được reviewer chấp nhận; không sửa annotation nguồn tại chỗ.
- Khóa decision bằng run, annotation ID và keypoint để đúng instance.
- Giữ `visible`, `occluded`, `outside`; đổi tọa độ không đổi state.
- Không tạo nhãn mới cho điểm `outside`.
- Giữ export hiện có, bổ sung temporal provenance.
- Báo cáo CSV/JSON chứa raw prediction, temporal estimate, algorithm/config version và review.

### 3.10. Video trực tiếp (Phase 7)

- Upload MP4/MOV/AVI cho Pose/Face nếu OpenCV đọc được; annotation tùy chọn; codec lỗi trả thông báo rõ.
- File tối đa 500 MB. Frame/decode quota chốt theo throughput thật; khởi đầu đề xuất 10.000 frame và 5 GB.
- Decode toàn bộ, giữ resolution, đánh số từ 0; chưa sampling FPS.
- Timestamp video hợp lệ được ưu tiên; fallback FPS và ghi source.
- Annotation map bằng original frame index, không suy đoán tên file.
- Import job: `processing`, `ready`, `failed`, `interrupted`; chỉ tạo QA run khi dataset `ready`.
- Startup chuyển job `processing` còn sót thành `interrupted`; retry tạo attempt mới.
- Decode vào staging; chỉ publish frame/dataset khi hoàn tất. Lỗi/quota phải cleanup staging, không tạo dataset `ready`.
- Mỗi video là một sequence; scene cut tạo segment.
- Pose nhận COCO Keypoints 1.0 và các JSON hiện có; Face giữ XML/JSON hiện có. Parser CVAT XML cho Pose chỉ thêm nếu Q6 cho thấy cần.
- Trước Phase 7, dữ liệu video được tách frame bằng script ngoài; script sinh luôn manifest (công cụ tạm).

### 3.11. Luồng chưa có annotation (Phase 8)

- Input: video (Phase 7) hoặc ZIP ảnh không nhãn; ZIP dùng cùng luồng suy sequence và xác nhận ở 3.8.
- Giữ luồng Face không nhãn hiện có, bổ sung temporal QC/provenance.
- Mở rộng luồng chưa nhãn cho Pose.
- Không dùng metric sai lệch annotation.
- Export `landmark-qa/v1` nhãn AI draft, kèm sequence/track, algorithm version và provenance.
- Visibility do AI tạo phải đánh dấu chưa xác nhận.

## 4. Các phase triển khai

Mỗi phase chạy regression suite hiện có và test của phase. S/M/L là ước lượng tương đối.

### Phase 0 — Chốt quyết định (S, không code)

- **Trạng thái: hoàn thành ngày 2026-10-16.** Q1–Q6 và bằng chứng được ghi ở mục 2; Q1–Q2 đã chốt tạm thời và còn tác vụ gửi mentor để xác nhận.
- **Điểm kiểm soát:** nếu mentor thay đổi Q1/Q2, cập nhật kế hoạch trước Phase 3; việc này không chặn Phase 1–2.

### Phase 1 — Metadata, suy sequence và xác nhận (M)

- **Trạng thái: hoàn thành ngày 2026-10-16.** Đã triển khai migration `003_temporal_metadata`, manifest/suy thứ tự/xác nhận sequence, bảo toàn source instance/track metadata, chặn temporal khi mapping chưa được xác nhận và bỏ mock fallback của Pose. Script video → frame là hạng mục tùy chọn nên để lại cho Phase 7.
- Migration sequence/image/run config và metadata source; backup, idempotent; backfill `import_order`.
- Tự suy sequence/thứ tự frame theo bảng ưu tiên ở 3.8; hiển thị delta/gap nhưng không tự kết luận sampling; manifest tùy chọn để ghi đè; API sequence summary, snapshot mapping và xác nhận (kèm FPS/sampling đều tùy chọn).
- Parser giữ frame/instance metadata; source track ID chỉ lưu nếu có.
- Dataset không suy ra được hoặc chưa xác nhận vẫn import nhưng không bật temporal.
- Bỏ mock fallback im lặng trong production Pose.
- Tùy chọn: script tách video → frame + manifest (công cụ tạm trước Phase 7).
- **Test:** migration hai lần; giữ data/review cũ; metadata xuyên import → DB; nhiều sequence; dataset cũ; suy thứ tự từ `file_name` (số có/không padding, nhiều prefix/numeric token); phân biệt COCO array order với numeric ID; gap đều/không đều không bị tự gán nguyên nhân; tên/nhóm mơ hồ bắt buộc manifest; sampled FPS chỉ áp dụng sau xác nhận; chưa xác nhận → temporal bị chặn; exception Pose tạo run failed, không mock warning.
- **Hoàn thành:** import nhiều sequence đúng metadata, xác nhận được qua API, không regression dataset cũ.

### Phase 2 — Ghép annotation, tracking và phân đoạn (M/L)

- **Trạng thái: hoàn thành ngày 2026-10-16.** Pose đã chuyển sang observation stream detection-first độc lập annotation; Pose/Face dùng matching một-một có ambiguity gate, track/segment nội bộ và lưu provenance trong `temporal_observations`. R13 Face chỉ nhận frame trước của cùng track đã xác nhận. Scene-cut và continuity threshold hiện là mặc định chưa hiệu chỉnh như rủi ro đã nêu.
- Tạo Pose observation stream độc lập annotation: primary `yolo26s-pose`, secondary-to-primary matching, ensemble `p_star` và provenance; mất primary thì ngắt track.
- Shared matcher không sửa matcher cũ: Pose Hungarian tâm + IoU; Face tái sử dụng cost/cutoff hiện tại và bổ sung ambiguity margin; mơ hồ → `tracking_uncertain`.
- Nối primary detection giữa các frame; scene segmentation; fast-path một người có continuity gate; R13 chỉ dùng confirmed track.
- **Test:** temporal observation không đổi khi annotation dịch chuyển hoặc vắng mặt; secondary ghép đúng primary; ensemble provenance/support count; mất primary ngắt track; hai annotation không cùng ghép một detection; ambiguity khi hai người sát/giao nhau; unmatched item được giữ; fast-path chỉ chạy khi đủ điều kiện; detection đổi thứ tự; scene cut/gap/đổi kích thước; matcher cũ không đổi baseline.
- **Hoàn thành:** matching một-một, không truyền lịch sử qua segment, mơ hồ thì bỏ qua hoặc tách track.

### Phase 3 — Temporal engine (M)

- **Trạng thái: hoàn thành ngày 2026-10-16.** Đã triển khai engine `robust-local-fit-v1`: cửa sổ 5–29, leave-one-out, normalization Pose/Face nhất quán, IRLS Huber bậc 2/bậc 1, interpolation/extrapolation, quality heuristic và đầy đủ unavailable reason. Gap dùng original frame delta và timestamp tin cậy; timestamp không hợp lệ hoặc chưa xác nhận fallback về frame index.
- Cửa sổ 5–29, normalization nhất quán, robust fit, leave-one-out, quality, gap/reason và config version.
- **Test:** 5/15/29; từ chối số chẵn/ngoài khoảng; đứng yên, đều, đổi hướng, outlier, đầu/cuối; < 7 không extrapolate; gap tính theo original frame index/time chứ không theo số ảnh import; sampled FPS đã/chưa xác nhận; timestamp lỗi; đổi normalization; thiếu target transform.
- **Hoàn thành:** mọi điểm trong phạm vi có estimate hoặc reason tái lập được.

### Phase 4 — QA/QC và API (M)

- **Trạng thái: hoàn thành ngày 2026-10-16.** Run snapshot `temporal_enabled`/window/version; temporal-off dùng nhánh QA/QC legacy. Temporal-on lưu kết quả mọi điểm, score/quality/evidence, `annotation_jitter`, unavailable reason và API results/detail/summary có phân trang, lọc, project authorization. R13 Face được đánh dấu `face_r13` và không tạo warning temporal trùng.
- Temporal result; inconsistency/jitter/score; create-run/results/detail/summary API; gộp evidence.
- `unavailable` là result status, không phải warning.
- **Test:** temporal off tương đương baseline; project authorization; run config bất biến; unavailable (kể cả `tracking_uncertain`) không vào warning queue; không đếm trùng.
- **Hoàn thành:** Pose/Face chạy temporal trên ZIP có annotation và truy vấn được API.

### Phase 5 — UI chỉ đọc (M/L) · Mốc demo 1

- **Trạng thái: hoàn thành ngày 2026-10-16.** Pose/Face có sequence summary và confirmation, upload manifest rời, toggle/window, summary/filter, overlay Human/Model/Temporal, trajectory, timeline thumbnail và navigation. Có [quickstart](TEMPORAL_QA_QUICKSTART.md); panel chỉ đọc, không thay đổi QA/QC cũ hay annotation nguồn.
- Chọn chế độ có nhãn/chưa nhãn, upload annotation rời, thẻ sequence summary, thumbnail đầu/cuối/quanh gap, ô FPS và xác nhận sampling đều; trường hợp mơ hồ yêu cầu manifest; toggle, window, overlay, trajectory, timeline, navigation, track/quality/source/reason và summary.
- README ngắn về chạy, input, manifest, config và giới hạn; frontend build.
- **Hoàn thành:** demo thả ZIP → xác nhận → temporal run → warning/trajectory → reason.

### Phase 6 — Chấp nhận điểm và export (M) · Mốc demo 2

- Chọn/đổi/bỏ temporal; lưu decision; export coordinate và audit report.
- **Test:** đúng keypoint/instance; giữ visibility; không tạo `outside`; export khớp decision/provenance.
- **Hoàn thành:** review và export đúng instance.

### Phase 7 — Upload video (L) · Mốc demo 3

- Upload, staging/decode nền, import job/quota/interruption/retry.
- **Test:** video hỏng; quá quota; lỗi giữa import; restart; rotation; codec lỗi; frame mapping; staging cleanup.
- **Hoàn thành:** lỗi không tạo dataset `ready`; restart không để job treo; map đúng frame.

### Phase 8 — Không annotation / AI draft (M) · Mốc demo 4

- Mở rộng Pose; temporal QC/provenance cho Face hiện có; ZIP ảnh không nhãn dùng luồng suy sequence; chuẩn hóa AI draft.
- **Test:** không sinh annotation-error metric; AI visibility chưa xác nhận; Face unlabeled không regression.
- **Hoàn thành:** Pose/Face chạy cả QA có nhãn và QC không nhãn.

### Điểm dừng an toàn

<table>
  <tr><th>Sau phase</th><th>Sản phẩm demo</th></tr>
  <tr><td>5</td><td>Temporal QA chỉ đọc trên ZIP có annotation, Pose + Face</td></tr>
  <tr><td>6</td><td>Thêm chấp nhận temporal và export</td></tr>
  <tr><td>7</td><td>Thêm upload video</td></tr>
  <tr><td>8</td><td>Thêm luồng chưa nhãn hoàn chỉnh</td></tr>
</table>

Nếu thiếu thời gian, dừng sau Phase 5 hoặc 6 vẫn có sản phẩm hoàn chỉnh.

## 5. Kiểm thử và dữ liệu tích hợp

- Regression backend, feature-boundary test và frontend build sau mỗi phase.
- Unit test temporal dùng chuỗi tổng hợp xác định, không phụ thuộc model/network.
- CI integration fixture được sinh xác định và commit cùng expected output. Smoke fixture ngoài repo theo Q4 phải có source URL, điều kiện sử dụng, frame range, ngày tải và SHA-256 trong fixture manifest trước khi chạy.
- Model inference mock trong CI; model thật chỉ ở smoke test/benchmark.
- Có fixture một người cho demo và nhiều người cho tracking/ghép annotation.
- Nghiệm thu chung: run cũ/annotation nguồn/QA hiện tại tiếp tục dùng; temporal off tương đương baseline; temporal 5–29 chạy Pose/Face; review/export đúng instance.

## 6. Rủi ro và lưu ý

- Chưa có benchmark temporal: không tuyên bố hiệu quả khôi phục điểm che; threshold mới đều chưa hiệu chỉnh.
- Chuyển động nhanh: fit và jitter không thay thế bằng chứng thật; quality thấp không tạo warning.
- MediaPipe thiếu confidence landmark: Face dựa matching, geometry, residual và ghi rõ confidence thiếu.
- Ghép annotation hoặc tracking sai có thể khiến so nhầm người và sinh cảnh báo giả: mơ hồ thì bỏ qua (`tracking_uncertain`), không ép ghép. Chấp nhận mất điểm hơn là báo sai.
- Thứ tự frame suy từ tên file/COCO có thể sai: hiển thị mapping, thumbnail và gap để xác nhận; trường hợp prefix/numeric token mơ hồ bắt buộc manifest. COCO array order và numeric ID là hai gợi ý khác nhau.
- Primary Pose mất detection sẽ ngắt track, làm giảm coverage ở đoạn che nặng; chấp nhận đánh đổi này để tránh ID switch trong bản đầu.
- Job không durable: Phase 7 chỉ phát hiện interruption và retry, chưa resume.
- Dataset cũ: `import_order` không đủ bật temporal.
- Dung lượng: lưu mọi result có thể lớn; API phải phân trang và xử lý theo sequence.

## 7. Kế hoạch tương lai (hoãn có chủ đích)

- Cập nhật tài liệu kiến trúc đầy đủ và liên kết từ `docs/README.md`.
- Benchmark temporal off/on theo video/sequence: precision/recall/FPR, lỗi coordinate visible/occluded/chuyển động nhanh, coverage, ID switch/fragmentation/refusal, thời gian, RAM, storage.
- Chỉ công bố hiệu quả phục hồi điểm che khi có ground truth hoặc reviewer độc lập.
- Hiệu chỉnh scene cut, gap, quality, jitter, margin ghép mơ hồ và warning threshold; quyết định rollout/scoring kết hợp.
- Mở rộng: secondary-model bridge qua đoạn primary mất detection với continuity gate chặt, temporal batch fix, realtime, learned model, Redis/Celery/resume, sampling FPS, re-identification, tạo `outside` mới, parser CVAT XML cho Pose (nếu cần) và downstream task theo mentor.
