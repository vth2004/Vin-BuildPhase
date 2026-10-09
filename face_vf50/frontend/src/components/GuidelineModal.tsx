import React, { useState, useMemo } from 'react';

export interface GuidelineRule {
  code: string;
  name: string;
  category: 'eye' | 'mouth' | 'nose' | 'eyebrow' | 'state' | 'schema';
  severity: 'critical' | 'major' | 'warning';
  glSection: string;
  points: string;
  shortDesc: string;
  businessLogic: string;
  mathCondition: string;
  cvatFix: string;
  inCabinContext: string;
}

export const GUIDELINE_RULES: GuidelineRule[] = [
  {
    code: 'R01',
    name: 'Đầy đủ cấu trúc 50 điểm',
    category: 'schema',
    severity: 'critical',
    glSection: 'Mục 1.3 & Mục 3, Mục 7.1',
    points: '0 – 49 (Đủ 7 skeletons)',
    shortDesc: 'Khuôn mặt bắt buộc có đủ 50 điểm cấu trúc từ 0 đến 49, không được xóa bất kỳ điểm nào.',
    businessLogic:
      'Bài nộp vi phạm thiếu điểm sẽ bị lỗi schema hệ thống CVAT/COCO và bị loại trực tiếp. 7 skeleton gồm: longmaytrai (0-4), longmayphai (5-9), songmui (10-13), mattrai (14-21), matphai (22-29), moingoai (30-41), moitrong (42-49).',
    mathCondition: '{i ∈ [0, 49]} ⊆ tập ID điểm gán nhãn. Nếu thiếu bất kỳ ID nào → Kích hoạt CRITICAL.',
    cvatFix:
      'Tìm skeleton bị thiếu điểm, thêm lại điểm với đúng ID quy định. Dù điểm bị che hoặc nằm ngoài ảnh, dùng trạng thái occluded hoặc outside thay vì xóa điểm.',
    inCabinContext: 'Thường xảy ra khi người gán lỡ tay bấm Delete một điểm khi cố chuyển trạng thái.',
  },
  {
    code: 'R02',
    name: 'Quên sửa trạng thái Visible',
    category: 'state',
    severity: 'warning',
    glSection: 'Mục 1.1 (#2), Mục 9',
    points: '0 – 49',
    shortDesc: 'Cảnh báo khi toàn bộ 50 điểm đều ở trạng thái Visible mặc dù có vật cản hoặc góc nghiêng.',
    businessLogic:
      'Hệ thống CVAT mặc định đặt sẵn 50 điểm ở trạng thái Visible. Nếu người gán nhãn không rà soát chuyển các điểm bị kính râm, gọng kính, tóc che sang Occluded hoặc Outside thì sẽ bị trừ điểm thẩm định chất lượng.',
    mathCondition: 'Số điểm có state == "visible" bằng 50/50 trên frame có yếu tố che khuất.',
    cvatFix:
      'Mở thuộc tính từng điểm (Properties) trên CVAT, chuyển các điểm bị che khuất sang occluded (hoặc phím tắt nếu có).',
    inCabinContext: 'Tài xế đeo kính mát, tóc mái rủ xuống trán hoặc ngón tay che miệng khi ngáp.',
  },
  {
    code: 'R03',
    name: 'Đảo Trái - Phải theo khung nhìn ảnh',
    category: 'schema',
    severity: 'critical',
    glSection: 'Mục 1.3 (Quy ước đối xứng)',
    points: '0-9, 14-29 (Lông mày & Mắt)',
    shortDesc: 'Quy ước Trái / Phải luôn theo góc nhìn người quan sát vào ảnh, không theo cơ thể người mẫu.',
    businessLogic:
      'Bên trái bức ảnh (hoành độ x nhỏ hơn) bắt buộc là longmaytrai (0-4) và mattrai (14-21). Bên phải bức ảnh (hoành độ x lớn hơn) bắt buộc là longmayphai (5-9) và matphai (22-29). Nhầm lẫn sẽ làm đảo lộn hoàn toàn bài thi.',
    mathCondition: 'mean(x_longmaytrai) < mean(x_longmayphai) VÀ mean(x_mattrai) < mean(x_matphai).',
    cvatFix:
      'Đổi tên nhãn hoặc hoán đổi tọa độ điểm giữa 2 cụm skeleton trái và phải trên CVAT.',
    inCabinContext: 'Lỗi rất phổ biến của người mới làm quen, hay nhầm với mắt trái của chính người mẫu.',
  },
  {
    code: 'R04',
    name: 'Thứ tự hoành độ x lông mày không tăng dần',
    category: 'eyebrow',
    severity: 'critical',
    glSection: 'Mục 2.2, Mục 7.1',
    points: '0-4 (Trái), 5-9 (Phải)',
    shortDesc: 'Hoành độ x của các điểm lông mày phải tăng dần từ trái sang phải ảnh.',
    businessLogic:
      'Quy chuẩn VinFast yêu cầu lông mày trái đánh số 0→4 từ đầu mày (trái) tới đuôi mày (phải). Lông mày phải đánh số 5→9 từ đầu mày (trái) tới đuôi mày (phải). Trục x phải đơn điệu tăng.',
    mathCondition: 'x[i] < x[i+1] đối với các cặp liên tiếp trong [0..4] và [5..9].',
    cvatFix:
      'Kéo và đổi lại đúng thứ tự các điểm trên cung mày theo chiều từ trái qua phải bức ảnh.',
    inCabinContext: 'Góc chụp nghiêng ¾ có thể làm đuôi mày co cụm, cần chú ý không để điểm sau vắt qua điểm trước.',
  },
  {
    code: 'R05',
    name: 'Khóe mắt cực trị không đúng',
    category: 'eye',
    severity: 'critical',
    glSection: 'Mục 2.2, Mục 7.4',
    points: '14, 18, 22, 26',
    shortDesc: 'Khóe mắt ngoài và khóe mắt trong phải giữ vị trí cực trị hoành độ x.',
    businessLogic:
      'Mắt trái: điểm 14 (khóe ngoài) phải là x nhỏ nhất, điểm 18 (khóe trong) là x lớn nhất. Mắt phải: điểm 22 (khóe trong) phải là x nhỏ nhất, điểm 26 (khóe ngoài) là x lớn nhất.',
    mathCondition: 'x[14] ≤ min(x_mattrai), x[18] ≥ max(x_mattrai), x[22] ≤ min(x_matphai), x[26] ≥ max(x_matphai).',
    cvatFix:
      'Đặt chính xác điểm 14 và 26 vào đuôi mắt ngoài; điểm 18 và 22 vào tuyến lệ/khóe mắt trong.',
    inCabinContext: 'Tránh để điểm mí trên hoặc mí dưới bị kéo phình sang ngang vượt qua khóe mắt.',
  },
  {
    code: 'R06',
    name: 'Mí trên thấp hơn mí dưới (Lật mí)',
    category: 'eye',
    severity: 'critical',
    glSection: 'Mục 5.1, 5.2, 7.4',
    points: '15-17 (mí trên T), 19-21 (mí dưới T), 23-25 (mí trên P), 27-29 (mí dưới P)',
    shortDesc: 'Bờ mí trên bị kéo tụt xuống thấp hơn bờ mí dưới (đặc biệt khi tài xế ngủ gật, nhắm mắt).',
    businessLogic:
      'Theo giải phẫu, mí trên luôn nằm cao hơn mí dưới (y mí trên nhỏ hơn y mí dưới do trục y hướng xuống). Ngay cả khi nhắm mắt, mí trên chỉ áp sát khe mi chứ không lộn ngược xuống dưới.',
    mathCondition: 'y\'_mí_trên ≤ y\'_mí_dưới + ε (sau khi đã khử góc nghiêng đầu θ theo đường nối 2 mắt).',
    cvatFix:
      'Kéo các điểm mí trên (15-17 hoặc 23-25) lên phía trên hoặc nằm sát ngay mép khe mi khép kín.',
    inCabinContext: 'Dữ liệu DMS phát hiện ngủ gật có nhiều ca mắt nhắm nghiền, rất dễ bị kéo đè lộn ngược mí.',
  },
  {
    code: 'R07',
    name: 'Mắt tự cắt chéo chữ X',
    category: 'eye',
    severity: 'critical',
    glSection: 'Mục 4.1, 7.4',
    points: '14-21 (Mắt trái), 22-29 (Mắt phải)',
    shortDesc: 'Đường viền khép kín của mắt tự đan chéo nhau tạo thành hình số 8 hoặc chữ X.',
    businessLogic:
      'Viền bờ mắt là một đa giác đơn giản khép kín (simple polygon). Nếu các đoạn thẳng cắt nhau, topology hình học bị đảo lộn hoàn toàn.',
    mathCondition: 'Tồn tại giao điểm giữa 2 cạnh không kề nhau của đa giác mắt.',
    cvatFix:
      'Rà soát thứ tự đi vòng của 8 điểm mắt: 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 để đảm bảo vòng tròn mở tự nhiên.',
    inCabinContext: 'Thường gặp khi nhắm mắt hoặc nháy mắt, người gán kéo vội điểm mí trên đan chéo mí dưới.',
  },
  {
    code: 'R08',
    name: 'Sống mũi chia không đều',
    category: 'nose',
    severity: 'major',
    glSection: 'Mục 2.2, Mục 5.3 & Hình 5',
    points: '10, 11, 12, 13 (songmui)',
    shortDesc: 'Ba đoạn sống mũi lệch nhau quá 1.45 lần, hoặc điểm 13 bị kéo xuống tận chóp mũi dưới.',
    businessLogic:
      'Sống mũi gồm 4 điểm: 10 (gốc mũi giữa 2 mắt), 11 & 12 (thân sống mũi), 13 (chân sống mũi ngang hai cánh mũi). Ba đoạn 10-11, 11-12, 12-13 phải chia tương đối đều nhau. Điểm 13 KHÔNG ĐƯỢC đặt ở đỉnh chóp mũi.',
    mathCondition: 'max(d1, d2, d3) / min(d1, d2, d3) > 1.45 với d1=|P10P11|, d2=|P11P12|, d3=|P12P13|.',
    cvatFix:
      'Dịch điểm 11 và 12 sao cho chia đều sống mũi thành 3 đoạn bằng nhau. Đảm bảo điểm 13 nằm trên đường ngang nối hai cánh mũi.',
    inCabinContext: 'Góc camera DMS từ trên trần xe nhìn xuống có thể làm sống mũi bị co ngắn quang học.',
  },
  {
    code: 'R09',
    name: 'Khóe môi trong lệch ngoài bờ môi ngoài',
    category: 'mouth',
    severity: 'critical',
    glSection: 'Mục 5.4, 6.4, 7.5',
    points: '30-41 (moingoai), 42-49 (moitrong)',
    shortDesc: 'Các điểm bờ môi trong bị kéo trồi ra ngoài vùng bao của môi ngoài.',
    businessLogic:
      'Về giải phẫu, toàn bộ 8 điểm môi trong (42-49) mô tả khoang miệng hở, bắt buộc phải nằm hoàn toàn trong lòng của 12 điểm môi ngoài (30-41).',
    mathCondition: 'Point-in-Polygon test: Điểm P ∈ moitrong nằm ngoài đa giác moingoai.',
    cvatFix:
      'Kéo các điểm môi trong lùi vào bên trong lòng môi. Nếu ngậm miệng, xếp 8 điểm đè sát lên đường tiếp giáp 2 môi.',
    inCabinContext: 'Xảy ra khi tài xế há to miệng ngáp, la hét hoặc nói chuyện điện thoại.',
  },
  {
    code: 'R10',
    name: 'Ngưỡng ẩn cả bộ phận (Mục 4.2 GL)',
    category: 'state',
    severity: 'critical',
    glSection: 'Mục 4.2 Guideline VinFast v1.3',
    points: 'Mắt (< 4/8 điểm) hoặc Lông mày (< 3/5 điểm)',
    shortDesc: 'Nếu một bộ phận bị che khuất quá nửa, BẮT BUỘC phải đánh Outside cho TOÀN BỘ các điểm của bộ phận đó.',
    businessLogic:
      'Quy tắc nghiêm ngặt của VinFast: Nếu một mắt chỉ nhìn thấy dưới 4/8 điểm (hoặc lông mày thấy dưới 3/5 điểm), không được cố chấm vài điểm nhìn thấy mà bắt buộc phải chuyển toàn bộ các điểm của bộ phận đó sang trạng thái Outside.',
    mathCondition: 'visible_count(mắt) < 4 nhưng còn điểm != outside, hoặc visible_count(mày) < 3 nhưng còn điểm != outside.',
    cvatFix:
      'Chọn tất cả các điểm của mắt hoặc lông mày đó trên CVAT và đổi trạng thái sang outside.',
    inCabinContext: 'Góc quay đầu lớn (Yaw > 45°) khiến nửa mặt khuất hẳn, hoặc bị tay/vô lăng che khuất phần lớn mắt.',
  },
  {
    code: 'R11',
    name: 'Điểm ngoài biên ảnh',
    category: 'schema',
    severity: 'critical',
    glSection: 'Mục 4.1',
    points: '0 – 49',
    shortDesc: 'Điểm rơi ra ngoài giới hạn ảnh (x < 0, x > W, y < 0, y > H) nhưng vẫn để Visible.',
    businessLogic:
      'Mọi điểm ngoài biên ảnh bắt buộc phải có trạng thái outside để hệ thống AI bỏ qua không tính loss.',
    mathCondition: '(x < 0 || x > 1280 || y < 0 || y > 720) && state != "outside".',
    cvatFix:
      'Chuyển trạng thái điểm sang outside, hoặc kéo lại vào trong ảnh nếu do lỡ tay kéo văng ra ngoài mép.',
    inCabinContext: 'Tài xế ngồi sát mép góc camera hoặc ảnh bị crop sát đầu.',
  },
  {
    code: 'R12',
    name: 'Trùng tọa độ bất thường',
    category: 'schema',
    severity: 'critical',
    glSection: 'Mục 1.3',
    points: '0 – 49',
    shortDesc: 'Hai điểm có ID khác nhau nhưng lại bị đặt đè lên cùng 1 tọa độ.',
    businessLogic:
      'Mỗi điểm mốc giải phẫu đại diện cho một vị trí riêng biệt. Trùng tọa độ chứng tỏ người chấm chưa tách điểm hoặc copy paste lỗi.',
    mathCondition: 'euclidean_dist(P_i, P_j) < 0.5px với i ≠ j.',
    cvatFix:
      'Tìm và kéo tách 2 điểm ra đúng vị trí giải phẫu riêng biệt của từng điểm.',
    inCabinContext: 'Thường xảy ra khi dùng phím tắt duplicate nhãn trên CVAT.',
  },
  {
    code: 'R13',
    name: 'Trôi điểm bất thường giữa 2 frame (Jitter)',
    category: 'schema',
    severity: 'major',
    glSection: 'Mục 6.9 (Video temporal consistency)',
    points: '0 – 49',
    shortDesc: 'Tọa độ cùng một điểm bị nhảy vọt quá lớn giữa 2 frame liên tiếp (> 15px scale-adaptive).',
    businessLogic:
      'Trong video cabin xe, khuôn mặt chuyển động liên tục. Nhảy vọt đột ngột thể hiện nhãn bị giật khung hình hoặc nhầm ID.',
    mathCondition: '||P_i(t+1) - P_i(t)|| > 15px × (IOD / 96.0).',
    cvatFix:
      'Kiểm tra lại 2 frame liền kề trên CVAT, căn chỉnh vị trí điểm để đảm bảo chuyển động trơn tru.',
    inCabinContext: 'Xe chạy qua gờ giảm tốc hoặc tài xế quay đầu đột ngột.',
  },
  {
    code: 'R14',
    name: 'Điểm Occluded giữ tọa độ sai giải phẫu',
    category: 'state',
    severity: 'warning',
    glSection: 'Mục 4.2 & Mục 8',
    points: '0 – 49',
    shortDesc: 'Điểm Occluded (kính râm, tóc) nhưng người gán ước lượng lệch quá 15% IOD.',
    businessLogic:
      'Mục 4.2 yêu cầu: Dù bị che bởi kính râm hoặc tóc, người gán vẫn phải ước lượng vị trí giải phẫu tự nhiên, không được kéo điểm đặt đè lên gọng kính hoặc ra ngoài mặt.',
    mathCondition: 'state == "occluded" VÀ dist(p_human, p_model) > 0.15 × IOD.',
    cvatFix:
      'Căn lại điểm occluded theo cấu trúc giải phẫu ước lượng của mắt sau kính hoặc dưới lọn tóc.',
    inCabinContext: 'Tài xế đeo kính râm bản to, mắt kính màu đen che khuất hoàn toàn con ngươi.',
  },
];

interface GuidelineModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedRuleCode?: string | null;
  onSelectRuleFilter?: (ruleCode: string) => void;
}

export const GuidelineModal: React.FC<GuidelineModalProps> = ({
  isOpen,
  onClose,
  selectedRuleCode,
  onSelectRuleFilter,
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [activeRuleCode, setActiveRuleCode] = useState<string>(selectedRuleCode || 'R01');

  // Sync when selectedRuleCode changes
  React.useEffect(() => {
    if (selectedRuleCode) {
      setActiveRuleCode(selectedRuleCode);
      const rule = GUIDELINE_RULES.find((r) => r.code === selectedRuleCode);
      if (rule) {
        setActiveCategory('all');
      }
    }
  }, [selectedRuleCode]);

  const filteredRules = useMemo(() => {
    return GUIDELINE_RULES.filter((rule) => {
      const matchCat = activeCategory === 'all' || rule.category === activeCategory;
      const q = searchQuery.toLowerCase().trim();
      const matchSearch =
        !q ||
        rule.code.toLowerCase().includes(q) ||
        rule.name.toLowerCase().includes(q) ||
        rule.shortDesc.toLowerCase().includes(q) ||
        rule.glSection.toLowerCase().includes(q);
      return matchCat && matchSearch;
    });
  }, [activeCategory, searchQuery]);

  const currentRule = useMemo(() => {
    return (
      GUIDELINE_RULES.find((r) => r.code === activeRuleCode) ||
      filteredRules[0] ||
      GUIDELINE_RULES[0]
    );
  }, [activeRuleCode, filteredRules]);

  if (!isOpen) return null;

  return (
    <div className="guideline-modal-overlay" onClick={onClose}>
      <div
        className="guideline-modal-container"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="guideline-modal-title"
      >
        {/* Modal Header */}
        <div className="guideline-modal-header">
          <div className="guideline-modal-header-left">
            <div className="guideline-modal-icon">📖</div>
            <div>
              <h2 id="guideline-modal-title" className="guideline-modal-title">
                Sổ Tay Quy Chuẩn Guideline VinFast VF-50 (14 Quy Tắc)
              </h2>
              <p className="guideline-modal-subtitle">
                Căn cứ chính thức: <strong>Week2_Guideline_Face_Landmark_VF50_HocVien_v1.3.docx</strong> — VinFast AI Thực Chiến
              </p>
            </div>
          </div>
          <button className="guideline-modal-close" onClick={onClose} title="Đóng cửa sổ">
            ✕
          </button>
        </div>

        {/* Modal Body: 2 Columns */}
        <div className="guideline-modal-body">
          {/* Left Column: Rule Selector & Categories */}
          <div className="guideline-sidebar">
            {/* Search Box */}
            <div className="guideline-search-box">
              <span className="search-icon">🔍</span>
              <input
                type="text"
                placeholder="Tìm quy tắc (R01, kính, mí, sống mũi...)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="guideline-search-input"
              />
              {searchQuery && (
                <button
                  className="search-clear-btn"
                  onClick={() => setSearchQuery('')}
                  title="Xóa tìm kiếm"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Category Filter Pills */}
            <div className="guideline-category-pills">
              {[
                { id: 'all', label: 'Tất cả (14)' },
                { id: 'eye', label: 'Mắt (Eye)' },
                { id: 'mouth', label: 'Môi (Mouth)' },
                { id: 'nose', label: 'Mũi (Nose)' },
                { id: 'eyebrow', label: 'Lông mày' },
                { id: 'state', label: 'Trạng thái' },
                { id: 'schema', label: 'Cấu trúc' },
              ].map((cat) => (
                <button
                  key={cat.id}
                  className={`guideline-cat-btn ${activeCategory === cat.id ? 'active' : ''}`}
                  onClick={() => setActiveCategory(cat.id)}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            {/* List of Rules */}
            <div className="guideline-rule-list">
              {filteredRules.length === 0 ? (
                <div className="guideline-empty">Không tìm thấy quy tắc phù hợp</div>
              ) : (
                filteredRules.map((rule) => {
                  const isSelected = rule.code === currentRule.code;
                  return (
                    <div
                      key={rule.code}
                      className={`guideline-rule-item ${isSelected ? 'selected' : ''}`}
                      onClick={() => setActiveRuleCode(rule.code)}
                    >
                      <div className="guideline-rule-item-top">
                        <span className="guideline-rule-code">{rule.code}</span>
                        <span className={`guideline-sev-badge ${rule.severity}`}>
                          {rule.severity.toUpperCase()}
                        </span>
                      </div>
                      <div className="guideline-rule-item-name">{rule.name}</div>
                      <div className="guideline-rule-item-ref">{rule.glSection}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Right Column: Full Rule Details */}
          <div className="guideline-detail-panel">
            {currentRule ? (
              <div className="guideline-detail-content">
                {/* Header of selected rule */}
                <div className="guideline-detail-header">
                  <div className="guideline-detail-title-wrap">
                    <span className="guideline-detail-code-badge">{currentRule.code}</span>
                    <h3 className="guideline-detail-name">{currentRule.name}</h3>
                  </div>
                  <div className="guideline-detail-meta">
                    <span className={`guideline-sev-badge ${currentRule.severity} large`}>
                      Mức độ: {currentRule.severity.toUpperCase()}
                    </span>
                    <span className="guideline-meta-item">
                      📍 <strong>Căn cứ:</strong> {currentRule.glSection}
                    </span>
                    <span className="guideline-meta-item">
                      🎯 <strong>Điểm ảnh hưởng:</strong> {currentRule.points}
                    </span>
                  </div>
                </div>

                {/* Short summary callout */}
                <div className="guideline-callout-summary">
                  <div className="callout-icon">💡</div>
                  <div className="callout-text">{currentRule.shortDesc}</div>
                </div>

                {/* Section: Bối cảnh Cabin & Bản chất nghiệp vụ */}
                <div className="guideline-section">
                  <h4 className="guideline-section-title">
                    <span className="section-bullet">1</span> Bản Chất Nghiệp Vụ & Bối Cảnh Cabin Xe
                  </h4>
                  <p className="guideline-text">{currentRule.businessLogic}</p>
                  <div className="guideline-incabin-box">
                    <strong>🚗 Tình huống DMS thực tế:</strong> {currentRule.inCabinContext}
                  </div>
                </div>

                {/* Section: Điều kiện toán học kiểm tra */}
                <div className="guideline-section">
                  <h4 className="guideline-section-title">
                    <span className="section-bullet">2</span> Điều Kiện Logic & Công Thức Toán Học
                  </h4>
                  <div className="guideline-code-block">{currentRule.mathCondition}</div>
                </div>

                {/* Section: Hướng dẫn sửa trên CVAT */}
                <div className="guideline-section">
                  <h4 className="guideline-section-title">
                    <span className="section-bullet">3</span> Hướng Dẫn Thao Tác Khắc Phục Trên CVAT
                  </h4>
                  <div className="guideline-fix-box">
                    <div className="fix-icon">🛠️</div>
                    <div className="fix-content">
                      <p>{currentRule.cvatFix}</p>
                    </div>
                  </div>
                </div>

                {/* Actions Footer */}
                <div className="guideline-detail-footer">
                  {onSelectRuleFilter && (
                    <button
                      className="guideline-action-filter-btn"
                      onClick={() => {
                        onSelectRuleFilter(currentRule.code);
                        onClose();
                      }}
                    >
                      🔍 Lọc các khung hình vi phạm [{currentRule.code}] ngay
                    </button>
                  )}
                  <button className="guideline-action-close-btn" onClick={onClose}>
                    Đóng sổ tay
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};
