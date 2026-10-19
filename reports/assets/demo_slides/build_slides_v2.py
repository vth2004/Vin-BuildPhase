# -*- coding: utf-8 -*-
"""Build 14 demo-day slides v2: HTML -> PNG (Chrome headless) -> PDF.

Cải tiến so với v1:
- 14 slide (trước 5), làm rõ bài toán và pipeline.
- Chèn ảnh demo thật của dự án (pose/face/temporal/UI).
- Bổ sung số liệu benchmark đầy đủ (BENCHMARK_REPORT.md + báo cáo OKS).
"""
import pathlib, subprocess
from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = pathlib.Path(r"D:\Vin\VinPrj\Vin-BuildPhase")
OUT = ROOT / "reports" / "assets" / "demo_slides"
OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path(r"C:\Users\hiepvt\AppData\Local\Temp\opencode\slides_work")
WORK.mkdir(parents=True, exist_ok=True)

def uri(p):
    return pathlib.Path(p).as_uri()

# Ảnh demo thật của dự án
PH_DRIV   = ROOT / "part_nghia" / "images" / "default" / "20130529_01_Driv_174_ll.jpg"
PH_CAR2   = ROOT / "part_nghia" / "images" / "default" / "pexels-photo-10400706.jpg"
PH_FACE   = ROOT / "deploy-data" / "projects" / "675de94f-8bac-4ae1-857f-96e13560545a" \
            / "datasets" / "37a809c4-ae30-483c-92d4-c1442f5b5a51" / "images" / "frame_000045.jpg"
FRAMES    = [ROOT / "deploy-data" / "projects" / "675de94f-8bac-4ae1-857f-96e13560545a"
             / "datasets" / "37a809c4-ae30-483c-92d4-c1442f5b5a51" / "images" / f"frame_0000{n:02d}.jpg"
             for n in (10, 30, 50, 70)]
SHOT_UI   = ROOT / "backend" / "tmp" / "docker-smoke" / "home.png"

TOTAL = 14

CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;overflow:hidden;background:#0a0b0e;color:#fafafa;
  font-family:'Segoe UI','Plus Jakarta Sans',system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
.slide{position:relative;width:1920px;height:1080px;padding:80px 96px 0;overflow:hidden;display:flex;flex-direction:column}
.grid{position:absolute;inset:0;background-image:
  linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:80px 80px;
  -webkit-mask-image:radial-gradient(115% 85% at 28% 32%,#000 18%,transparent 74%);
  mask-image:radial-gradient(115% 85% at 28% 32%,#000 18%,transparent 74%)}
.glow{position:absolute;border-radius:50%;filter:blur(150px)}
.g1{width:760px;height:760px;right:-200px;top:-260px;background:#ea580c;opacity:.22}
.g2{width:620px;height:620px;left:-240px;top:160px;background:#2563eb;opacity:.16}
.g3{width:560px;height:560px;right:60px;bottom:-320px;background:#059669;opacity:.14}
.head{position:relative;z-index:2;flex:0 0 auto}
.body{position:relative;z-index:2;flex:1;display:flex;flex-direction:column;justify-content:center;padding-bottom:86px;min-height:0}
.kicker{font-size:19px;font-weight:700;letter-spacing:.26em;text-transform:uppercase;color:#a1a1aa;display:flex;align-items:center;gap:16px}
.kicker .bar{width:44px;height:3px;border-radius:2px;background:linear-gradient(90deg,#ea580c,#f59e0b)}
h1.title{font-size:68px;font-weight:800;letter-spacing:-.032em;line-height:1.06;margin-top:18px}
h1.title .accent{background:linear-gradient(92deg,#fb923c,#f59e0b 55%,#fbbf24);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{font-size:24px;color:#a1a1aa;margin-top:14px;line-height:1.5;max-width:1360px}
.sub b{color:#e4e4e7;font-weight:600}
.footer{position:absolute;left:96px;right:96px;bottom:38px;display:flex;justify-content:space-between;
  align-items:center;font-size:17px;color:#6b6b73;font-weight:600;letter-spacing:.05em;z-index:2}
.footer b{color:#d4d4d8;letter-spacing:.08em}
.card{background:linear-gradient(180deg,#12141a,#0d0f14);border:1px solid rgba(255,255,255,.09);
  border-radius:20px;box-shadow:0 28px 60px rgba(0,0,0,.5)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.o{color:#fb923c}.g{color:#34d399}.b{color:#93c5fd}.r{color:#f87171}.w{color:#fafafa}
/* --- photos --- */
.shot{position:relative;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.14);
  background:#000;box-shadow:0 30px 70px rgba(0,0,0,.55)}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(5,6,9,.12),rgba(5,6,9,.62) 82%)}
.shot .cap{position:absolute;left:20px;right:20px;bottom:16px;display:flex;align-items:center;gap:12px;
  font-size:18px;font-weight:700;color:#e9e9ee}
.shot .cap .badge{margin-left:auto;padding:5px 12px;border-radius:8px;font-size:14px;font-weight:700;
  background:rgba(10,12,16,.72);border:1px solid rgba(255,255,255,.16);color:#d4d4d8}
.shot .hud{position:absolute;left:18px;top:18px;width:290px;padding:14px 16px;border-radius:13px;
  background:rgba(13,15,20,.88);border:1px solid rgba(255,255,255,.12)}
.shot .hud .k{font-size:13px;color:#8b8b93;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.shot .hud .row{margin-top:10px;display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700}
.shot .hud .row .v{margin-left:auto}
.tag{padding:4px 10px;border-radius:7px;font-size:14px}
.to{background:rgba(249,115,22,.18);color:#fb923c;border:1px solid rgba(249,115,22,.4)}
.tg{background:rgba(16,185,129,.16);color:#34d399;border:1px solid rgba(16,185,129,.4)}
.tb{background:rgba(59,130,246,.18);color:#93c5fd;border:1px solid rgba(59,130,246,.4)}
/* --- kpi --- */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-top:40px}
.kpi{padding:26px 26px 22px}
.kpi .v{font-size:52px;font-weight:800;letter-spacing:-.03em}
.kpi .l{margin-top:8px;font-size:17px;color:#a1a1aa;line-height:1.4}
.kpi small{display:block;margin-top:4px;font-size:14px;color:#7c7c85}
/* --- pipeline flow --- */
.pipe{display:flex;align-items:stretch;gap:12px}
.stepf{flex:1;padding:20px 18px;border-radius:14px;background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.11);text-align:center}
.stepf .n{font-size:14px;font-weight:800;color:#8b8b93;letter-spacing:.1em}
.stepf .t{margin-top:8px;font-size:20px;font-weight:800;color:#e4e4e7}
.stepf .s{margin-top:6px;font-size:15px;color:#8b8b93}
.stepf.cur{background:linear-gradient(180deg,rgba(249,115,22,.14),rgba(249,115,22,.06));
  border-color:rgba(249,115,22,.5)}
.stepf.cur .t{color:#fdba74}
.arrow{align-self:center;color:#52525b;font-size:24px;font-weight:700;flex:0 0 auto}
/* --- bullets --- */
.bul{display:flex;flex-direction:column;gap:20px}
.bul li{list-style:none;display:flex;gap:15px;font-size:22px;line-height:1.42;color:#c4c4c9}
.bul li b{color:#fafafa;font-weight:700}
.mk{flex:0 0 10px;height:10px;border-radius:3px;background:#f97316;margin-top:8px}
.mk.g{background:#34d399}.mk.b{background:#93c5fd}.mk.r{background:#f87171}
.idx{flex:0 0 34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:800;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);color:#e4e4e7}
.idx.g{background:rgba(16,185,129,.16);border-color:rgba(16,185,129,.35);color:#34d399}
.idx.o{background:rgba(249,115,22,.16);border-color:rgba(249,115,22,.35);color:#fb923c}
/* --- tables --- */
table.tbl{width:100%;border-collapse:collapse;font-size:19px}
.tbl th{text-align:left;font-size:14px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;
  color:#8b8b93;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.14)}
.tbl td{padding:12px 14px;border-bottom:1px solid rgba(255,255,255,.07);color:#d4d4d8;font-weight:600}
.tbl td.num,.tbl th.num{text-align:right}
.tbl tr.hi td{background:linear-gradient(90deg,rgba(249,115,22,.14),rgba(249,115,22,.05));color:#fdba74}
.tbl td small{display:block;font-size:13px;color:#7c7c85;font-weight:600;margin-top:2px}
/* --- bars --- */
.charts{display:flex;flex-direction:column;gap:26px}
.mrow{display:grid;grid-template-columns:250px 1fr 130px;align-items:center;gap:18px}
.mname{font-size:20px;font-weight:700;color:#e4e4e7}
.mname small{display:block;font-size:13px;font-weight:600;color:#7c7c85;letter-spacing:.05em;text-transform:uppercase;margin-top:3px}
.bars{display:flex;flex-direction:column;gap:7px}
.bar{height:27px;border-radius:6px;display:flex;align-items:center;padding-left:12px;font-size:15px;font-weight:700;color:#0a0b0e}
.bar.k1{background:#3f3f46;color:#e4e4e7}
.bar.up{background:linear-gradient(90deg,#ea580c,#f59e0b)}
.bar.down{background:linear-gradient(90deg,#059669,#10b981)}
.bar.weak{background:linear-gradient(90deg,#9a3412,#c2410c)}
.delta{font-size:26px;font-weight:800;text-align:right}
.track{height:14px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden}
.fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#34d399,#10b981)}
.fill.o{background:linear-gradient(90deg,#ea580c,#f59e0b)}
.rl{display:grid;grid-template-columns:160px 1fr 70px;gap:14px;align-items:center;font-size:17px;font-weight:700;color:#d4d4d8;white-space:nowrap}
.notes{display:flex;gap:40px;font-size:19px;color:#c4c4c9}
.notes span{display:flex;align-items:center;gap:12px}
.note{margin-top:26px;font-size:18px;color:#7c7c85;line-height:1.55;max-width:1600px}
.note b{color:#c4c4c9}
.framestrip{display:flex;gap:18px;align-items:stretch}
.frm{flex:1;position:relative;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,.14)}
.frm img{width:100%;height:100%;object-fit:cover;display:block}
.frm .fn{position:absolute;left:10px;top:10px;padding:4px 10px;border-radius:7px;font-size:13px;font-weight:800;
  background:rgba(10,12,16,.75);border:1px solid rgba(255,255,255,.16);color:#d4d4d8}
.fsp{align-self:center;color:#f59e0b;font-size:22px;font-weight:800}
.pillset{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}
.pillset span{font-size:15px;font-weight:700;color:#d4d4d8;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);border-radius:999px;padding:5px 12px}
/* --- side list slide cuối --- */
.side{display:flex;gap:60px;align-items:flex-start}
.side .col{flex:1}
.side h3{font-size:25px;font-weight:800;letter-spacing:.02em;margin:0 0 24px;display:flex;align-items:center;gap:12px}
.side ul{list-style:none;display:flex;flex-direction:column;gap:22px}
.side li{display:flex;gap:16px;font-size:21px;line-height:1.4;color:#c4c4c9}
.side li b{color:#fafafa;font-weight:700}
"""

SKELETON = r"""
<svg viewBox="0 0 700 560" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%">
  <g transform="translate(249,44) scale(1.05)">
    <g stroke="#52525b" stroke-width="2.6" stroke-linecap="round" fill="none">
      <circle cx="210" cy="48" r="26"/>
      <path d="M210 76v26"/><path d="M210 102 122 124"/><path d="M210 102 298 124"/>
      <path d="M120 124h180"/><path d="M120 124 96 192"/><path d="M300 124 324 192"/>
      <path d="M96 192 110 262"/><path d="M324 192 310 262"/><path d="M210 102 204 242"/>
      <path d="M204 242 150 252"/><path d="M204 242 262 252"/><path d="M120 124 150 252"/>
      <path d="M300 124 262 252"/><path d="M150 252 136 352"/><path d="M262 252 280 352"/>
      <path d="M136 352 146 442"/><path d="M280 352 270 442"/>
    </g>
    <g fill="#a1a1aa" opacity="0.8">
      <circle cx="120" cy="124" r="4.4"/><circle cx="96" cy="192" r="4.4"/><circle cx="310" cy="262" r="4.4"/>
      <circle cx="150" cy="252" r="4.4"/><circle cx="262" cy="252" r="4.4"/><circle cx="136" cy="352" r="4.4"/>
      <circle cx="280" cy="352" r="4.4"/><circle cx="146" cy="442" r="4.4"/><circle cx="270" cy="442" r="4.4"/>
      <circle cx="210" cy="48" r="4.4"/><circle cx="210" cy="102" r="4.4"/><circle cx="110" cy="262" r="4.4"/>
    </g>
    <circle cx="334" cy="206" r="6.6" fill="#059669"/>
    <circle cx="334" cy="206" r="13" fill="none" stroke="#10b981" stroke-width="2" opacity="0.55"/>
    <circle cx="324" cy="192" r="7.2" fill="#f97316"/>
    <circle cx="324" cy="192" r="17" fill="none" stroke="#f97316" stroke-width="2.4" opacity="0.7"/>
    <path d="M324 192 L334 206" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="3 4"/>
  </g>
</svg>
"""

def footer(idx):
    return (f'<div class="footer"><span><b>LANDMARK QA/QC</b> &nbsp;·&nbsp; ĐỘI 005 · WAR</span>'
            f'<span>{idx:02d} / {TOTAL:02d}</span></div>')

def shell(body):
    return ("<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body><div class='slide'>"
            "<div class='grid'></div><div class='glow g1'></div><div class='glow g2'></div><div class='glow g3'></div>"
            f"{body}</div></body></html>")

def head(k, t, s=""):
    sub = f'<p class="sub">{s}</p>' if s else ""
    return f'<div class="head"><div class="kicker"><span class="bar"></span> {k}</div><h1 class="title">{t}</h1>{sub}</div>'

def pipe(steps, cur):
    html = ""
    for i, (t, s) in enumerate(steps):
        if i: html += '<span class="arrow">→</span>'
        c = " cur" if i == cur else ""
        html += f'<div class="stepf{c}"><div class="n">BƯỚC {i+1}</div><div class="t">{t}</div><div class="s">{s}</div></div>'
    return f'<div class="pipe">{html}</div>'

STEPS = [("ZIP / video", "dữ liệu vào"), ("AI inference", "Pose + Face"),
         ("Ghép &amp; đối chiếu", "annotation ↔ detection"), ("Chấm điểm", "nghi lỗi + xếp hạng"),
         ("Temporal QA/QC", "track xuyên frame"), ("Review &amp; Export", "nhãn sạch + audit")]

slides = []

# ============================================================ 1 · COVER
s = f"""
<div class="body" style="justify-content:center">
<div style="display:flex;align-items:center;gap:44px;flex:1">
  <div style="flex:1;max-width:900px">
    <div class="kicker"><span class="bar"></span> VinFast Data Quality · Demo Day</div>
    <div style="font-size:112px;font-weight:800;letter-spacing:-.045em;line-height:.98;margin-top:26px">
      LANDMARK<br/><span class="accent" style="background:linear-gradient(92deg,#fb923c,#f59e0b 55%,#fbbf24);-webkit-background-clip:text;background-clip:text;color:transparent">QA/QC</span>
    </div>
    <p style="font-size:28px;line-height:1.5;color:#a1a1aa;margin-top:32px;max-width:820px">
      Kiểm định nhãn keypoint bằng <b style="color:#fafafa">AI Ensemble</b> và <b style="color:#fafafa">luật hình học</b>.<br/>
      Chỉ ra điểm nghi lỗi để reviewer quyết định nhanh — thay vì soi 100% bằng mắt.
    </p>
    <div style="margin-top:42px;display:flex;align-items:center;gap:18px">
      <span style="padding:12px 24px;border-radius:999px;font-size:21px;font-weight:700;letter-spacing:.06em;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.14)">ĐỘI 005 · WAR</span>
      <span style="font-size:20px;color:#71717a;font-weight:600">Human Pose 17 &nbsp;·&nbsp; Face Landmark VF50 &nbsp;·&nbsp; Temporal QA/QC</span>
    </div>
    <div style="margin-top:36px;display:flex;gap:34px;font-size:19px;color:#a1a1aa">
      <span>✔ Ensemble <b class="w">↓53% báo động oan</b></span>
      <span>✔ Scoring OKS <b class="w">AUROC 0.916</b></span>
      <span>✔ 14 luật hình học <b class="w">F1 tới 1.00</b></span>
    </div>
  </div>
  <div style="flex:0 0 640px;height:720px" class="card">
    <div style="height:100%;position:relative;border-radius:20px;overflow:hidden;background:radial-gradient(80% 80% at 62% 45%,rgba(59,130,246,.10),transparent 72%),#080a0d">
      {SKELETON}
      <div class="hud" style="position:absolute;left:18px;top:18px;width:290px;padding:14px 16px;border-radius:13px;background:rgba(13,15,20,.88);border:1px solid rgba(255,255,255,.12)">
        <div style="font-size:13px;color:#8b8b93;font-weight:700;letter-spacing:.07em;text-transform:uppercase">Cảnh báo · frame 12 · kp 10</div>
        <div style="margin-top:10px;display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700"><span class="tag to">swap trái/phải</span><span class="v" style="margin-left:auto">0.92</span></div>
        <div style="margin-top:10px;display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700"><span class="tag tg">Temporal Q</span><span class="v" style="margin-left:auto">87%</span></div>
      </div>
    </div>
  </div>
</div>
</div>
{footer(1)}
"""
slides.append(shell(s))

# ============================================================ 2 · BÀI TOÁN & CHI PHÍ
s = f"""
{head("Bài toán", "Nhãn bẩn là <span class='accent'>nợ kỹ thuật</span> có lãi kép",
      "Gán nhãn keypoint là nền của mọi model pose/face. Lỗi không được phát hiện lúc gán sẽ truyền sâu vào model — và càng về sau càng đắt để sửa.")}
<div class="body">
<div style="display:flex;gap:26px;align-items:stretch;min-height:0">
  <div class="card" style="flex:1.25;padding:30px 32px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">QUY TRÌNH HIỆN TẠI — KHÔNG CÓ KHÂU KIỂM ĐỊNH</div>
    <div style="display:flex;flex-direction:column;gap:13px;margin-top:22px">
      <div style="display:flex;align-items:center;gap:14px;font-size:21px;font-weight:700"><span class="idx">1</span> Annotator gán 17–50 điểm/khung bằng tay</div>
      <div style="margin-left:16px;color:#52525b;font-size:20px">↓</div>
      <div style="display:flex;align-items:center;gap:14px;font-size:21px;font-weight:700"><span class="idx">2</span> Đi thẳng vào tập train — <b class="r">không ai rà soát từng điểm</b></div>
      <div style="margin-left:16px;color:#52525b;font-size:20px">↓</div>
      <div style="display:flex;align-items:center;gap:14px;font-size:21px;font-weight:700"><span class="idx">3</span> Model học từ sai số hệ thống: nhầm trái/phải, điểm lơ lửng, nhảy frame</div>
      <div style="margin-left:16px;color:#52525b;font-size:20px">↓</div>
      <div style="display:flex;align-items:center;gap:14px;font-size:21px;font-weight:700"><span class="idx o">4</span> Phát hiện khi model đã suy giảm → <b class="o">gán lại toàn bộ — chi phí lãi kép</b></div>
    </div>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:20px;min-height:0">
    <div class="shot" style="flex:1.15;min-height:0">
      <img src="{uri(PH_DRIV)}"/><div class="shade"></div>
      <div class="cap">Dữ liệu thật trong dự án<span class="badge">17 điểm / khung</span></div>
      <div class="hud"><div class="k">Reviewer phải soi</div>
        <div class="row"><span class="tag to">17–50</span><span>điểm mỗi khung — bằng mắt</span></div>
        <div class="row"><span class="tag tb">100%</span><span>khung ảnh đều phải duyệt</span></div>
      </div>
    </div>
    <div class="card" style="padding:20px 26px;font-size:19px;line-height:1.5;color:#a1a1aa">
      <b class="w">Quy mô minh hoạ:</b> bộ face chuẩn WFLW chỉ có <b class="w">10.000 face × 98 điểm ≈ 1 triệu điểm</b> — soi bằng mắt là không thể mở rộng.
    </div>
  </div>
</div>
</div>
{footer(2)}
"""
slides.append(shell(s))

# ============================================================ 3 · BA LOẠI LỖI
s = f"""
{head("Bài toán · Phân loại lỗi", "Ba kiểu sai mà annotator hay mắc nhất")}
<div class="body">
<div class="cards3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:26px">
  <div class="card" style="padding:28px 30px">
    <div style="display:flex;align-items:center;gap:12px;font-size:26px;font-weight:800"><span class="dot" style="background:#f97316;width:12px;height:12px"></span>Lệch tọa độ</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.45;color:#a1a1aa">Điểm vẽ không khớp vị trí người trong ảnh — lơ lửng hoặc dính khớp cạnh.</p>
    <div class="pillset"><span>Dịch nhỏ: 5–10px</span><span>Dịch lớn: 25–45px</span><span>OKS phát hiện</span></div>
  </div>
  <div class="card" style="padding:28px 30px">
    <div style="display:flex;align-items:center;gap:12px;font-size:26px;font-weight:800"><span class="dot" style="background:#f87171;width:12px;height:12px"></span>Hoán đổi trái/phải</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.45;color:#a1a1aa">Swap tay, chân, mắt — model học sai hệ quy chiếu, lỗi khó thấy khi review nhanh.</p>
    <div class="pillset"><span>Cặp đối xứng L↔R</span><span>AUROC 0.958 bắt được</span></div>
  </div>
  <div class="card" style="padding:28px 30px">
    <div style="display:flex;align-items:center;gap:12px;font-size:26px;font-weight:800"><span class="dot" style="background:#38bdf8;width:12px;height:12px"></span>Trôi &amp; giật theo frame</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.45;color:#a1a1aa">Điểm nhảy bất thường giữa các frame liên tiếp — chỉ thấy khi nhìn chuỗi, không thấy trên 1 ảnh.</p>
    <div class="pillset"><span>Temporal window 5–29</span><span>Track xuyên frame</span></div>
  </div>
</div>
<div class="card" style="margin-top:30px;padding:24px 30px">
  <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">HỆ THỐNG BẮT ĐƯỢC TỪNG LOẠI NHƯ THẾ NÀO? — POSE, COCO TEST (BENCHMARK 2026-10)</div>
  <div style="display:flex;gap:44px;margin-top:14px;font-size:20px;font-weight:700;align-items:center;flex-wrap:wrap">
    <span>Dịch lớn <b class="g">AUROC 0.959</b></span>
    <span>Swap L/R <b class="g">AUROC 0.958</b></span>
    <span>Dịch nhỏ <b class="o">0.732</b></span>
    <span>Sai state occluded <b class="r">0.271</b> <small style="color:#7c7c85;font-weight:600">— điểm yếu đã ghi nhận</small></span>
  </div>
</div>
</div>
{footer(3)}
"""
slides.append(shell(s))

# ============================================================ 4 · OKS
def kpi(v, l, extra="", cls="w"):
    return f'<div class="card kpi"><div class="v {cls}">{v}</div><div class="l">{l}</div><small>{extra}</small></div>'

s = f"""
{head("Giải pháp · Scoring", "Chấm điểm theo <span class='accent'>giải phẫu</span>, không theo cảm tính",
      "Thay công thức tuyến tính cổ điển (dist_norm × 4.5) bằng phân phối hàm mũ chuẩn OKS — mỗi khớp có dung sai riêng theo COCO Keypoint.")}
<div class="body">
<div class="kpis">
  {kpi("0.782 → 0.916", "<b>AUROC</b> — phân định lỗi thật / điểm đúng", "+17.1%", "o")}
  {kpi("65% → 90%", "<b>Precision@20</b> — 20 cảnh báo đầu là lỗi thật", "+25%", "o")}
  {kpi("70% → 100%", "<b>Precision@10</b> — Top-10 không còn oan", "+30%", "g")}
  {kpi("38.2% → 12.1%", "Cảnh báo oan ở khớp <b>hông / vai</b>", "↓ 68.4%", "g")}
</div>
<div style="display:flex;gap:26px;margin-top:28px">
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">TẠI SAO CŨ BÁO OAN?</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Lệch <b class="w">16px ở hông</b> (bình thường — cha xương rộng) và lệch <b class="w">8px ở mắt</b> (nguy hiểm) bị phạt <b>cùng một mốc hệ số 4.5</b>. Kết quả: lỗi thật bị chôn dưới đống cảnh báo rác.</p>
  </div>
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">OKS LÀM GÌ KHÁC?</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Mắt/mũi σ≈0.025 → dung sai hẹp, lệch nhẹ là báo ngay. Hông/vai σ≈0.107 → dung sai rộng, hấp thụ rung tự nhiên. <b class="w">Chi phí tính thêm ≈ 0.001 ms/ảnh</b> — không cần thêm model.</p>
  </div>
</div>
</div>
{footer(4)}
"""
slides.append(shell(s))

# ============================================================ 5 · PIPELINE OVERVIEW
s = f"""
{head("Giải pháp", "Một pipeline — <span class='accent'>sáu khâu tự động</span>, hai chuẩn nhãn")}
<div class="body">
{pipe(STEPS, -1)}
<div style="display:flex;gap:26px;margin-top:52px">
  <div class="card" style="flex:1;padding:28px 30px">
    <div class="cap" style="display:flex;align-items:center;gap:12px;font-size:23px;font-weight:800"><span class="dot" style="background:#f97316;width:12px;height:12px"></span>Human Pose VF17</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Ensemble <b class="w">YOLO26s + RTMPose</b> hợp nhất thành vị trí đồng thuận; chấm điểm nghi ngờ theo chuẩn OKS.</p>
    <div class="pillset"><span>Ensemble K2</span><span>OKS σ</span><span>Swap L/R</span></div>
  </div>
  <div class="card" style="flex:1;padding:28px 30px">
    <div class="cap" style="display:flex;align-items:center;gap:12px;font-size:23px;font-weight:800"><span class="dot" style="background:#38bdf8;width:12px;height:12px"></span>Face Landmark VF50</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">MediaPipe 478 điểm ánh xạ về <b class="w">50 điểm</b>, kiểm định bằng <b class="w">14 luật hình học</b> R01–R14 và NME chuẩn hoá theo IOD.</p>
    <div class="pillset"><span>R01–R14</span><span>NME</span><span>Hungarian</span></div>
  </div>
  <div class="card" style="flex:1;padding:28px 30px">
    <div class="cap" style="display:flex;align-items:center;gap:12px;font-size:23px;font-weight:800"><span class="dot" style="background:#34d399;width:12px;height:12px"></span>Temporal QA/QC</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Nối cùng một người qua nhiều frame, hồi quy quỹ đạo để <b class="w">gợi ý điểm</b> và bắt nhãn nhảy bất thường.</p>
    <div class="pillset"><span>Track xuyên frame</span><span>Cửa sổ 5–29</span><span>Gợi ý điểm</span></div>
  </div>
</div>
<p class="note">Tất cả chạy <b>CPU-only, offline</b> — không cần GPU hay cloud. Phần sau đi từng khâu kèm ảnh demo thật của dự án.</p>
</div>
{footer(5)}
"""
slides.append(shell(s))

# ============================================================ 6 · BƯỚC 1 DỮ LIỆU VÀO
s = f"""
{head("Pipeline · Bước 1", "Nhận ảnh, ZIP và <span class='accent'>video</span> — trả kết quả theo từng frame")}
<div class="body">
<div style="display:flex;gap:24px;min-height:0">
  <div class="shot" style="flex:1.2"><img src="{uri(PH_DRIV)}"/><div class="shade"></div>
    <div class="cap">Ảnh đơn<span class="badge">.jpg / .png cả folder</span></div></div>
  <div class="shot" style="flex:1"><img src="{uri(PH_CAR2)}"/><div class="shade"></div>
    <div class="cap">ZIP dataset<span class="badge">batch nhiều ảnh</span></div></div>
  <div class="shot" style="flex:1"><img src="{uri(FRAMES[1])}"/><div class="shade"></div>
    <div class="cap">Video<span class="badge">frame_000XX.jpg</span></div></div>
</div>
<div class="bul" style="margin-top:32px;flex-direction:row;gap:44px">
  <li style="flex:1"><span class="mk"></span><span>Video được <b>tách frame tự động</b> — mỗi frame là một bản ghi riêng, giữ nguyên thứ tự thời gian để khâu Temporal nối quỹ đạo.</span></li>
  <li style="flex:1"><span class="mk g"></span><span>Người dùng nén ZIP cả <b>hàng trăm ảnh</b> — pipeline xử lý batch nền, không chặn UI.</span></li>
  <li style="flex:1"><span class="mk b"></span><span>Chuẩn nhãn hỗ trợ: <b>COCO-17</b> (thân thể) và <b>VF50</b> (7 vùng ngũ quan mặt) — đúng hai chuẩn nghiệp vụ đang dùng.</span></li>
</div>
</div>
{footer(6)}
"""
slides.append(shell(s))

# ============================================================ 7 · BƯỚC 2 POSE
s = f"""
{head("Pipeline · Bước 2", "AI Inference — <span class='accent'>Ensemble hai model</span> cho thân thể")}
<div class="body">
<div style="display:flex;gap:30px;min-height:0">
  <div class="shot" style="flex:1.25">
    <img src="{uri(PH_DRIV)}"/><div class="shade"></div>
    <div class="cap">Dữ liệu dự án — lái xe, người ngồi lệch góc<span class="badge">VF17</span></div>
    <div class="hud"><div class="k">Ensemble · frame demo</div>
      <div class="row"><span class="tag tb">YOLO26s-pose</span><span class="v" style="margin-left:auto">0.87</span></div>
      <div class="row"><span class="tag to">RTMPose-m</span><span class="v" style="margin-left:auto">0.91</span></div>
      <div class="row"><span class="tag tg">Đồng thuận OKS</span><span class="v" style="margin-left:auto">92%</span></div>
    </div>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:20px">
    <ul class="bul">
      <li><span class="mk"></span><span><b>Hai model độc lập</b> chấm cùng bộ điểm; hệ thống hợp nhất thành vị trí đồng thuận và đo độ bất đồng giữa chúng.</span></li>
      <li><span class="mk r"></span><span>Bất đồng lớn giữa 2 model + lệch so với annotation người + luật hình học vi phạm → <b>điểm nghi ngờ tăng</b> theo chuẩn OKS.</span></li>
      <li><span class="mk g"></span><span>Sanity check: tự đo AP đúng điều kiện công bố — <b>YOLO26s-pose 62.56</b> (công bố 63.0), <b>RTMPose-m 74.58</b> (74.6, lệch 0.021%) → pipeline đo đúng, không vòng vo.</span></li>
    </ul>
    <div class="card" style="padding:22px 26px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">MEDIAN THỜI GIAN / ẢNH — CPU</div>
      <div style="display:flex;gap:36px;margin-top:12px;font-size:20px;font-weight:700">
        <span>K1 (1 model) <b class="w">85.8 ms</b></span>
        <span>K2 (2 model) <b class="o">505 ms</b></span>
        <span style="color:#7c7c85;font-weight:600;font-size:17px">chọn K2: chất lượng QA quan trọng hơn latency</span>
      </div>
    </div>
  </div>
</div>
</div>
{footer(7)}
"""
slides.append(shell(s))

# ============================================================ 8 · BƯỚC 3 FACE
s = f"""
{head("Pipeline · Bước 3", "Face VF50 — <span class='accent'>14 luật hình học</span> + đo khoảng cách theo IOD")}
<div class="body">
<div style="display:flex;gap:30px;min-height:0">
  <div class="shot" style="flex:1.25">
    <img src="{uri(PH_FACE)}"/><div class="shade"></div>
    <div class="cap">Dữ liệu dự án — video tin tức, face góc nghiêng<span class="badge">VF50</span></div>
    <div class="hud"><div class="k">Kiểm định luật · demo</div>
      <div class="row"><span class="tag tb">MediaPipe 478 → 50</span></div>
      <div class="row"><span class="tag tg">R01 cấu trúc</span><span class="v" style="margin-left:auto">F1 1.00</span></div>
      <div class="row"><span class="tag to">R13 scale/trôi</span><span class="v" style="margin-left:auto">F1 0.99</span></div>
      <div class="row"><span class="tag tb">NME / IOD</span><span class="v" style="margin-left:auto">chuẩn hoá</span></div>
    </div>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:20px">
    <ul class="bul">
      <li><span class="mk b"></span><span>MediaPipe trả <b>478 điểm lưới dày</b>; ánh xạ về đúng <b>50 điểm mốc khuôn mặt</b> chuẩn VF50 — contour resample theo chiều dài cung, vùng ngoài ảnh để null.</span></li>
      <li><span class="mk"></span><span><b>Hai mapper</b> (WFLW-98 và iBUG-68 → VF50) ghép chuỗi bằng thuật toán <b>Hungarian</b> — nhãn trái/phải không bao giờ bị suy diễn bậy.</span></li>
      <li><span class="mk g"></span><span>Mọi sai số đều <b>chuẩn hoá theo IOD</b> (khoảng cách hai mắt) — so được giữa ảnh gần, ảnh xa, khác máy quay.</span></li>
    </ul>
    <div class="card" style="padding:22px 26px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">BASELINE MEDIAPIPE TRÊN DỮ LIỆU CÔNG KHAI</div>
      <div style="display:flex;gap:34px;margin-top:12px;font-size:20px;font-weight:700;flex-wrap:wrap">
        <span>300W <b class="w">NME 8.54%</b></span>
        <span>WFLW <b class="w">NME 9.14%</b></span>
        <span style="color:#7c7c85;font-weight:600;font-size:17px">lát khó nhất: blur NME 14.5%, pose NME 21.9% — chính là vùng cần QA</span>
      </div>
    </div>
  </div>
</div>
</div>
{footer(8)}
"""
slides.append(shell(s))

# ============================================================ 9 · BƯỚC 5 TEMPORAL
s = f"""
{head("Pipeline · Bước 5", "Temporal QA/QC — <span class='accent'>nhìn chuỗi frame</span>, không nhìn 1 ảnh")}
<div class="body">
<div class="framestrip">
  <div class="frm"><img src="{uri(FRAMES[0])}"/><div class="fn">t−10</div></div><span class="fsp">⋯</span>
  <div class="frm"><img src="{uri(FRAMES[1])}"/><div class="fn">t−5</div></div><span class="fsp">→</span>
  <div class="frm" style="border-color:rgba(249,115,22,.55)"><img src="{uri(PH_FACE)}"/><div class="fn" style="color:#fdba74">t — điểm nghi lỗi</div></div><span class="fsp">→</span>
  <div class="frm"><img src="{uri(FRAMES[2])}"/><div class="fn">t+5</div></div><span class="fsp">⋯</span>
  <div class="frm"><img src="{uri(FRAMES[3])}"/><div class="fn">t+10</div></div>
</div>
<div style="display:flex;gap:26px;margin-top:36px">
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">TRACK XUYÊN FRAME</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Nối cùng một người qua các frame, hồi quy quỹ đạo từng điểm để <b class="w">gợi ý vị trí đúng</b> khi annotator chấm sai.</p>
    <div class="pillset"><span>Cửa sổ 5–29 frame</span><span>Hồi quy quỹ đạo</span><span>Gợi ý điểm 1-click</span></div>
  </div>
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">BẮT NHÃN NHẢY</div>
    <p style="margin-top:12px;font-size:20px;line-height:1.5;color:#a1a1aa">Click gật giữa frame t−1 → t → t+1 vì <b class="w">vượt ngưỡng chuyển động tự nhiên</b> sẽ được đánh dấu — loại lỗi chỉ tồn tại trong video.</p>
    <div class="pillset"><span>Frame t−1 · t · t+1</span><span>Ngưỡng chuyển động</span><span>Temporal Q 0–100%</span></div>
  </div>
</div>
</div>
{footer(9)}
"""
slides.append(shell(s))

# ============================================================ 10 · BƯỚC 4+6 REVIEW UI
s = f"""
{head("Pipeline · Bước 4 & 6", "Reviewer <span class='accent'>duyệt điểm nghi lỗi</span>, máy làm phần còn lại",
      "Hệ thống xếp hạng từng điểm theo độ nghi ngờ — reviewer duyệt Top-K cảnh báo, không quét toàn bộ." )}
<div class="body">
<div style="display:flex;gap:30px;align-items:stretch;min-height:0">
  <div class="shot" style="flex:1.5;display:flex;align-items:center;justify-content:center;background:#f5f5f4">
    <img src="{uri(SHOT_UI)}" style="object-fit:contain;object-position:top"/>
    <div class="cap" style="color:#3f3f46">Giao diện thật của hệ thống — hai studio Pose &amp; Face<span class="badge" style="color:#3f3f46;border-color:rgba(0,0,0,.2)">web app · v2.4</span></div>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:20px">
    <ul class="bul">
      <li><span class="mk"></span><span>Hệ thống <b>xếp hạng từng điểm</b> theo độ nghi ngờ — reviewer làm việc với <b class="o">Top-K cảnh báo</b> thay vì quét toàn bộ.</span></li>
      <li><span class="mk b"></span><span>Review từng điểm: <b>chấp nhận / đổi nhãn / bỏ qua</b>, có lịch sử thao tác và log kiểm định để đối soát về sau.</span></li>
      <li><span class="mk g"></span><span>Xuất <b>nhãn sạch kèm audit log</b> — mỗi chỉnh sửa đều truy vết được người, thời gian, lý do.</span></li>
    </ul>
    <div class="card" style="padding:22px 26px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">GIÁ TRỊ THỰC ĐO — SCORING OKS</div>
      <div style="display:flex;gap:34px;margin-top:12px;font-size:20px;font-weight:700;flex-wrap:wrap">
        <span>P@10 <b class="g">100%</b></span>
        <span>P@20 <b class="g">90%</b></span>
        <span>P@50 <b class="o">78%</b></span>
        <span style="color:#7c7c85;font-weight:600;font-size:17px">Duyệt Top-50: tỷ lệ "duyệt oan" giảm từ 46% xuống 22% — hơn một nửa công vô ích bị loại bỏ</span>
      </div>
    </div>
  </div>
</div>
</div>
{footer(10)}
"""
slides.append(shell(s))

# ============================================================ 11 · BENCH POSE
def mrow(name, sub, before, after, bw, aw, cls, delta, dcls):
    return (f'<div class="mrow"><div class="mname">{name}<small>{sub}</small></div>'
            f'<div class="bars"><div class="bar k1" style="width:{bw}%">K1 {before}</div>'
            f'<div class="bar {cls}" style="width:{aw}%">K2 {after}</div></div>'
            f'<div class="delta {dcls}">{delta}</div></div>')

s = f"""
{head("Benchmark · Pose", "Ensemble (K2) <span class='accent'>giảm báo động oan</span> trên COCO TEST",
      "500 ảnh COCO val2017 · chia DEV/TEST seed 42 · ngưỡng khoá trên DEV trước khi chạm TEST · CI 95% bootstrap 1.000 lần theo ảnh.")}
<div class="body">
<div style="display:flex;gap:44px">
  <div class="charts" style="width:980px;gap:38px">
    {mrow("AUPRC", "chất lượng xếp hạng", "0.190", "0.283", 67, 100, "up", "+49%", "o")}
    {mrow("Precision", "độ chính xác cảnh báo", "0.244", "0.389", 63, 100, "up", "+60%", "o")}
    {mrow("False alarm", "FPR — thấp hơn tốt", "5.76%", "2.72%", 100, 47, "down", "−53%", "g")}
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:18px">
    <div class="card" style="padding:24px 28px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">BẮT TỪNG LOẠI LỖI (AUROC K1 → K2)</div>
      <table class="tbl" style="margin-top:10px">
        <tr><th>Loại lỗi</th><th class="num">K1</th><th class="num">K2</th></tr>
        <tr><td>Dịch lớn</td><td class="num">0.947</td><td class="num">0.959</td></tr>
        <tr><td>Swap trái–phải</td><td class="num">0.942</td><td class="num">0.958</td></tr>
        <tr><td>Dịch nhỏ</td><td class="num">0.700</td><td class="num">0.732</td></tr>
        <tr><td>Sai state occluded <small>hạn chế đã ghi nhận</small></td><td class="num">0.303</td><td class="num r">0.271</td></tr>
      </table>
    </div>
    <div class="card" style="padding:20px 28px;font-size:18px;line-height:1.5;color:#a1a1aa">
      <b class="w">K3 (3 model)?</b> Chỉ +0.013 AUPRC so với K2 nhưng thêm ~415 ms và ~307 MiB — <b class="w">không đáng chi phí</b>, loại khỏi phạm vi triển khai.
    </div>
  </div>
</div>
</div>
{footer(11)}
"""
slides.append(shell(s))

# ============================================================ 12 · BENCH TỔNG QUÁT + KỶ LUẬT
s = f"""
{head("Benchmark · Tổng quát", "Xu hướng <span class='accent'>giữ nguyên</span> ngoài COCO — và mọi con số có dây chuyền kiểm")}
<div class="body">
<div style="display:flex;gap:26px">
  <div class="card" style="flex:1.1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">CROWDPOSE VAL — 500/2.000 ẢNH</div>
    <div class="charts" style="gap:24px;margin-top:18px">
      <div class="mrow" style="grid-template-columns:150px 1fr 110px;gap:12px">
        <div class="mname">AUPRC<small>K1 → K2</small></div>
        <div class="bars"><div class="bar k1" style="width:60%">K1 0.103</div><div class="bar up" style="width:100%">K2 0.173</div></div>
        <div class="delta o">+68%</div>
      </div>
      <div class="mrow" style="grid-template-columns:150px 1fr 110px;gap:12px">
        <div class="mname">False alarm<small>FPR — thấp hơn tốt</small></div>
        <div class="bars"><div class="bar k1" style="width:100%">K1 14.5%</div><div class="bar down" style="width:45%">K2 5.3%</div></div>
        <div class="delta g">−63%</div>
      </div>
    </div>
    <p style="margin-top:18px;font-size:17px;color:#7c7c85;line-height:1.5">Dataset đông người, khó hơn — K2 vẫn thắng K1 trên mọi metric. Ngưỡng tối ưu theo miền dữ liệu.</p>
  </div>
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">NGUYÊN TẮC KHOÁ NGƯỠNG</div>
    <ul class="bul" style="margin-top:16px">
      <li><span class="mk"></span><span>Ngưỡng <b>chỉ chọn trên DEV</b>, khoá cứng, rồi đánh giá <b>một lần duy nhất trên TEST</b> — không tune trên TEST.</span></li>
      <li><span class="mk g"></span><span>CrowdPose chọn phía DEV ra ngưỡng 0.20 khác COCO 0.30 → <b>phản bác cách áp một ngưỡng chung</b> cho mọi miền dữ liệu.</span></li>
      <li><span class="mk o"></span><span>Kết quả: <b>giữ nguyên ngưỡng production 0.20</b> — benchmark chưa đủ cơ sở đổi đồng loạt.</span></li>
    </ul>
  </div>
  <div class="card" style="flex:1;padding:26px 30px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">AP SANITY — PIPELINE ĐO ĐÚNG</div>
    <table class="tbl" style="margin-top:12px">
      <tr><th>Model</th><th class="num">Công bố</th><th class="num">Tự đo</th></tr>
      <tr><td>YOLO26s-pose</td><td class="num">AP 63.0</td><td class="num">62.56</td></tr>
      <tr><td>RTMPose-m</td><td class="num">AP 74.6</td><td class="num">74.58</td></tr>
    </table>
    <p style="margin-top:14px;font-size:17px;color:#7c7c85;line-height:1.5">Sai lệch 0.021% so với model zoo chính thức → phần cứng/cấu hình của benchmark đã đúng, số QA bên trên đáng tin.</p>
  </div>
</div>
</div>
{footer(12)}
"""
slides.append(shell(s))

# ============================================================ 13 · BENCH FACE
def rule_row(code, f1, width, weak=False):
    c = " fill o" if weak else " fill"
    return (f'<div class="rl"><span>{code}</span>'
            f'<div class="track"><div class="{c}" style="width:{width}%"></div></div>'
            f'<span>{f1}</span></div>')

s = f"""
{head("Benchmark · Face VF50", "14 luật hình học: <span class='accent'>mạnh đạt F1 1.00</span>, yếu lộ diện rõ",
      "1.300 face TEST độc lập (WFLW + 300W) · tiêm 39.000 lỗi có kiểm soát · control sạch để đo báo oan.")}
<div class="body">
<div style="display:flex;gap:40px">
  <div class="card" style="flex:1.05;padding:28px 32px">
    <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">F1 THEO LUẬT — TEST</div>
    <div style="display:flex;flex-direction:column;gap:16px;margin-top:20px">
      {rule_row("R01 · cấu trúc", "1.000", 100)}
      {rule_row("R10 · mắt/mày", "1.000", 100)}
      {rule_row("R14 · trôi", "1.000", 100)}
      {rule_row("R13 · scale", "0.993", 99)}
      {rule_row("R11 · đối xứng", "0.936", 94)}
      {rule_row("R03 · hình học", "0.912", 91)}
      {rule_row("R08 · tỷ lệ", "0.767", 77)}
      {rule_row("R05 · vùng", "0.622", 62, True)}
      {rule_row("R04 · scale eye", "0.601", 60, True)}
      {rule_row("R07 / R09 / R12", "0.36–0.42", 40, True)}
    </div>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;gap:18px">
    <div class="card" style="padding:24px 28px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">ĐIỂM NGHI NGỜ THEO MỨC NHIỄU — AUROC</div>
      <div class="charts" style="gap:16px;margin-top:16px">
        <div class="mrow"><div class="mname">Nhiễu 1% IOD<small>rất tinh vi</small></div><div class="bars"><div class="bar up" style="width:72%">0.737</div></div><div></div></div>
        <div class="mrow"><div class="mname">5%<small>lỗi người gán điển hình</small></div><div class="bars"><div class="bar up" style="width:85%">0.854</div></div><div></div></div>
        <div class="mrow"><div class="mname">10%<small>rõ ràng</small></div><div class="bars"><div class="bar down" style="width:96%">0.958</div></div><div></div></div>
        <div class="mrow"><div class="mname">20%<small>nhãn sai nặng</small></div><div class="bars"><div class="bar down" style="width:99%">0.985</div></div><div></div></div>
      </div>
    </div>
    <div class="card" style="padding:22px 28px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">TRANH THẬT CHỮ — BÁO SAI TRÊN VIDEO THẬT</div>
      <p style="margin-top:12px;font-size:19px;line-height:1.55;color:#a1a1aa">R13 trên 2.990 cặp frame 300VW còn <b class="o">FPR 10.6%</b> ở video khó. Recall 0.99 ở bảng bên trái đo trên chuỗi tổng hợp có jump biết trước — <b class="w">hai con số trả lời hai câu hỏi khác nhau, không tráo đổi.</b></p>
    </div>
  </div>
</div>
</div>
{footer(13)}
"""
slides.append(shell(s))

# ============================================================ 14 · KHẢ THI & TIẾP THEO
def li14(i, text, cls="", tag="✓"):
    return f'<li><span class="idx {cls}">{tag}</span><span>{text}</span></li>'

s = f"""
{head("Khả thi · Bước tiếp theo", "Đã dùng được — và con đường <span class='accent'>đo bằng nhãn thật</span>")}
<div class="body">
<div class="side">
  <div class="col">
    <h3><span class="dot" style="background:#34d399"></span>Hiện có trong demo này</h3>
    <ul>
      {li14(0, "<b>CPU-only, offline</b> — không cần GPU hay cloud", "g")}
      {li14(0, "<b>Nhận ảnh, ZIP, video</b> — kết quả theo từng frame", "g")}
      {li14(0, "<b>Ba tầng kiểm định</b>: OKS scoring · 14 luật face · temporal", "g")}
      {li14(0, "<b>Review từng điểm</b> + xuất nhãn sạch với audit log", "g")}
    </ul>
    <div class="card" style="margin-top:30px;padding:24px 28px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">SỐ LIỆU NÀY LẤY Ở ĐÂU?</div>
      <p style="margin-top:10px;font-size:18px;line-height:1.5;color:#a1a1aa">Benchmark tái lập được: chỉ dùng dataset công khai <b class="w">COCO val2017 · CrowdPose · WFLW · 300W · 300VW</b> — không dùng dữ liệu VinFast, raw evidence có SHA-256 checksum.</p>
    </div>
  </div>
  <div class="col">
    <h3><span class="dot" style="background:#fb923c"></span>Bước tiếp theo</h3>
    <ul>
      {li14(1, "Tập <b>lỗi gán nhãn thật</b> do người adjudicate độc lập — khoá làm TEST thực địa", "o", "1")}
      {li14(2, "<b>Hiệu chỉnh R02/R10/R14</b> khi có dữ liệu visibility/occlusion thật (COFW-68, MERL-RAV)", "o", "2")}
      {li14(3, "<b>Pilot downstream</b>: nhãn sạch giúp model train tốt hơn bao nhiêu — so 3 kịch bản nhiễu/QA sạch", "o", "3")}
      {li14(4, "<b>Cân bằng tốc độ</b>: chạy K2 bất đồng bộ, hoặc chỉ kích hoạt khi K1 có uncertainty cao", "o", "4")}
    </ul>
    <div class="card" style="margin-top:30px;padding:24px 28px">
      <div style="font-size:14px;font-weight:800;letter-spacing:.1em;color:#8b8b93">LỜI NGỎ</div>
      <p style="margin-top:10px;font-size:20px;line-height:1.5;color:#c4c4c9">Mỗi giờ reviewer không phải soi điểm oan là <b class="o">một giờ tìm được lỗi thật</b>. Hệ thống đưa chúng ta đến đúng chỗ đó.</p>
    </div>
  </div>
</div>
</div>
{footer(14)}
"""
slides.append(shell(s))

# ============================================================ render
profile = WORK / "chrome-profile"
pngs = []
for i, html in enumerate(slides, 1):
    hp = WORK / f"slide{i}.html"
    hp.write_text(html, encoding="utf-8")
    op = OUT / f"slide{i}.png"
    if op.exists():
        op.unlink()
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--no-first-run", "--no-default-browser-check",
                    f"--user-data-dir={profile}", "--force-device-scale-factor=1",
                    "--window-size=1920,1080", "--virtual-time-budget=4500",
                    f"--screenshot={op}", "file:///" + str(hp).replace("\\", "/")],
                   capture_output=True)
    print("rendered", op.name, op.stat().st_size)
    pngs.append(op)

imgs = [Image.open(p).convert("RGB") for p in pngs]
pdf = OUT / "LandmarkQA_QC_slides_v2.pdf"
imgs[0].save(pdf, "PDF", resolution=144.0, save_all=True, append_images=imgs[1:])
print("PDF:", pdf, pdf.stat().st_size)
