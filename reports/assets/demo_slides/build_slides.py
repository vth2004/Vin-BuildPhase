# -*- coding: utf-8 -*-
"""Build 5 demo-day slides: HTML -> PNG (Chrome headless) -> PDF."""
import os, subprocess, tempfile, pathlib
from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = pathlib.Path(r"D:\Vin\VinPrj\Vin-BuildPhase\reports\assets\demo_slides")
OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path(r"C:\Users\hiepvt\AppData\Local\Temp\opencode\slides_work")
WORK.mkdir(parents=True, exist_ok=True)

CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;overflow:hidden;background:#0a0b0e;color:#fafafa;
  font-family:'Segoe UI','Plus Jakarta Sans',system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
.slide{position:relative;width:1920px;height:1080px;padding:86px 96px 0;overflow:hidden;display:flex;flex-direction:column}
.head{flex:0 0 auto}
.body{flex:1;display:flex;flex-direction:column;justify-content:center;padding-bottom:92px}
.grid{position:absolute;inset:0;background-image:
  linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:80px 80px;
  -webkit-mask-image:radial-gradient(115% 85% at 28% 32%,#000 18%,transparent 74%);
  mask-image:radial-gradient(115% 85% at 28% 32%,#000 18%,transparent 74%)}
.glow{position:absolute;border-radius:50%;filter:blur(150px)}
.g1{width:760px;height:760px;right:-200px;top:-260px;background:#ea580c;opacity:.24}
.g2{width:620px;height:620px;left:-240px;top:160px;background:#2563eb;opacity:.18}
.g3{width:560px;height:560px;right:60px;bottom:-320px;background:#059669;opacity:.15}
.head{position:relative;z-index:2}
.kicker{font-size:19px;font-weight:700;letter-spacing:.26em;text-transform:uppercase;color:#a1a1aa;display:flex;align-items:center;gap:16px}
.kicker .bar{width:44px;height:3px;border-radius:2px;background:linear-gradient(90deg,#ea580c,#f59e0b)}
h1.title{font-size:74px;font-weight:800;letter-spacing:-.032em;line-height:1.04;margin-top:20px}
h1.title .accent{background:linear-gradient(92deg,#fb923c,#f59e0b 55%,#fbbf24);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{font-size:25px;color:#a1a1aa;margin-top:16px;line-height:1.5;max-width:1280px}
.sub b{color:#e4e4e7;font-weight:600}
.footer{position:absolute;left:96px;right:96px;bottom:40px;display:flex;justify-content:space-between;
  align-items:center;font-size:18px;color:#6b6b73;font-weight:600;letter-spacing:.05em;z-index:2}
.footer b{color:#d4d4d8;letter-spacing:.08em}
.card{background:linear-gradient(180deg,#12141a,#0d0f14);border:1px solid rgba(255,255,255,.09);
  border-radius:20px;box-shadow:0 28px 60px rgba(0,0,0,.5)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.o{color:#fb923c}.g{color:#34d399}.b{color:#93c5fd}.w{color:#fafafa}
"""

SKELETON = r"""
<svg viewBox="0 0 700 560" preserveAspectRatio="xMidYMid meet" class="sk">
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

FACE = r"""
<svg viewBox="0 0 320 340" preserveAspectRatio="xMidYMid meet">
  <g stroke="#3f3f46" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M160 34c-55 0-91 41-91 101v33c0 60 37 115 91 139 54-24 91-79 91-139v-33c0-60-36-101-91-101Z"/>
    <path d="M69 155c-15-15-20 14-4 37m186-37c15-15 20 14 4 37"/>
    <path d="M84 119c19-16 44-18 67-5m18 0c23-13 48-11 67 5"/>
    <path d="M88 146c15-13 39-15 55 0-16 11-40 11-55 0Zm89 0c16-15 40-13 55 0-15 11-39 11-55 0Z"/>
    <path d="M160 123v55l-13 20c7 8 19 8 26 0l-13-20"/>
    <path d="M113 218c16-11 33-12 47-6 14-6 31-5 47 6-11 13-28 20-47 20s-36-7-47-20Z"/>
    <path d="M135 220c15-5 35-5 50 0-15 8-35 8-50 0Z"/>
  </g>
  <g fill="#a1a1aa" opacity="0.75">
    <circle cx="84" cy="119" r="2.6"/><circle cx="100" cy="110" r="2.6"/><circle cx="118" cy="106" r="2.6"/><circle cx="136" cy="108" r="2.6"/><circle cx="151" cy="114" r="2.6"/>
    <circle cx="169" cy="114" r="2.6"/><circle cx="184" cy="108" r="2.6"/><circle cx="202" cy="106" r="2.6"/><circle cx="220" cy="110" r="2.6"/><circle cx="236" cy="119" r="2.6"/>
    <circle cx="160" cy="123" r="2.6"/><circle cx="160" cy="142" r="2.6"/><circle cx="160" cy="161" r="2.6"/><circle cx="160" cy="178" r="2.6"/>
    <circle cx="88" cy="146" r="2.6"/><circle cx="100" cy="139" r="2.6"/><circle cx="116" cy="136" r="2.6"/><circle cx="132" cy="139" r="2.6"/><circle cx="143" cy="146" r="2.6"/>
    <circle cx="132" cy="152" r="2.6"/><circle cx="116" cy="155" r="2.6"/><circle cx="100" cy="152" r="2.6"/>
    <circle cx="177" cy="146" r="2.6"/><circle cx="188" cy="139" r="2.6"/><circle cx="204" cy="136" r="2.6"/><circle cx="220" cy="139" r="2.6"/><circle cx="232" cy="146" r="2.6"/>
    <circle cx="220" cy="152" r="2.6"/><circle cx="204" cy="155" r="2.6"/><circle cx="188" cy="152" r="2.6"/>
    <circle cx="113" cy="218" r="2.6"/><circle cx="125" cy="211" r="2.6"/><circle cx="139" cy="207" r="2.6"/><circle cx="151" cy="210" r="2.6"/><circle cx="160" cy="212" r="2.6"/>
    <circle cx="169" cy="210" r="2.6"/><circle cx="181" cy="207" r="2.6"/><circle cx="195" cy="211" r="2.6"/><circle cx="207" cy="218" r="2.6"/>
  </g>
  <circle cx="204" cy="136" r="5" fill="#f97316"/><circle cx="204" cy="136" r="9.5" fill="none" stroke="#f97316" stroke-width="2" opacity="0.6"/>
  <circle cx="139" cy="207" r="5" fill="#059669"/><circle cx="139" cy="207" r="9.5" fill="none" stroke="#10b981" stroke-width="2" opacity="0.6"/>
</svg>
"""

EXTRA_CSS = r"""
.sk{width:100%;height:100%;display:block}
/* slide 1 */
.title-wrap{display:flex;align-items:center;gap:40px;flex:1}
.title-left{flex:1;max-width:920px}
.big{font-size:118px;font-weight:800;letter-spacing:-.045em;line-height:.98}
.big .accent{background:linear-gradient(92deg,#fb923c,#f59e0b 55%,#fbbf24);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:29px;line-height:1.5;color:#a1a1aa;margin-top:34px;max-width:840px}
.lead b{color:#fafafa;font-weight:600}
.team{margin-top:46px;display:flex;align-items:center;gap:18px}
.team .pill{padding:12px 24px;border-radius:999px;font-size:22px;font-weight:700;letter-spacing:.06em;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.14);color:#fafafa}
.team .muted{font-size:21px;color:#71717a;font-weight:600}
.hero{position:relative;width:700px;height:760px;border-radius:26px;overflow:hidden;
  background:radial-gradient(80% 80% at 62% 45%,rgba(59,130,246,.10),transparent 72%),#080a0d;
  border:1px solid rgba(255,255,255,.08);box-shadow:0 40px 90px rgba(0,0,0,.55)}
.hero .hud{position:absolute;left:18px;top:18px;width:300px;padding:15px 17px;border-radius:14px;
  background:rgba(17,19,24,.85);border:1px solid rgba(255,255,255,.1)}
.hud .k{font-size:14px;color:#8b8b93;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.hud .row{margin-top:11px;display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700}
.tag{padding:5px 11px;border-radius:8px;font-size:15px}
.to{background:rgba(249,115,22,.16);color:#fb923c;border:1px solid rgba(249,115,22,.35)}
.tg{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.35)}
.hud .row .v{margin-left:auto}
/* slide 2/3/5 grids */
.cards3{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:54px}
.pcard{padding:30px 32px;position:relative}
.pcard h3{margin:20px 0 0;font-size:28px;font-weight:800;letter-spacing:-.01em}
.pcard p{margin:12px 0 0;font-size:21px;line-height:1.45;color:#a1a1aa}
.thumb{height:200px;border-radius:14px;background:#080a0d;border:1px solid rgba(255,255,255,.07);display:flex;align-items:center;justify-content:center;overflow:hidden}
.thumb svg{width:100%;height:100%}
.notes{display:flex;gap:44px;margin-top:48px;font-size:24px;color:#c4c4c9}
.notes span{display:flex;align-items:center;gap:14px}
.notes .mk{width:10px;height:10px;border-radius:3px;background:#f97316;flex:0 0 10px}
/* pipeline */
.pipe{display:flex;align-items:center;gap:14px;margin-top:56px;flex-wrap:nowrap}
.step{padding:26px 24px;border-radius:14px;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.11);
  font-size:21px;font-weight:700;color:#e4e4e7;text-align:center;flex:1}
.step small{display:block;margin-top:8px;font-size:16px;font-weight:600;color:#8b8b93;letter-spacing:.04em}
.arrow{color:#52525b;font-size:26px;font-weight:700;flex:0 0 auto}
/* slide 3 feature cards */
.fcard{padding:34px 34px 30px}
.fcard .cap{display:flex;align-items:center;gap:12px;font-size:24px;font-weight:800;letter-spacing:-.01em}
.fcard .cap .dot{width:12px;height:12px}
.fcard p{margin:16px 0 0;font-size:21px;line-height:1.5;color:#a1a1aa}
.fcard .kw{margin-top:20px;display:flex;gap:9px;flex-wrap:wrap}
.fcard .kw span{font-size:16px;font-weight:700;color:#d4d4d8;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);border-radius:999px;padding:5px 12px}
/* slide 4 chart */
.chart{margin-top:26px;display:flex;flex-direction:column;gap:44px;width:1080px}
.mrow{display:grid;grid-template-columns:230px 1fr 210px;align-items:center;gap:22px}
.mname{font-size:23px;font-weight:700;color:#e4e4e7}
.mname small{display:block;font-size:15px;font-weight:600;color:#7c7c85;letter-spacing:.05em;text-transform:uppercase;margin-top:4px}
.bars{display:flex;flex-direction:column;gap:9px}
.bar{height:32px;border-radius:7px;display:flex;align-items:center;padding-left:14px;font-size:17px;font-weight:700;color:#0a0b0e}
.bar.k1{background:#3f3f46;color:#e4e4e7}
.bar.up{background:linear-gradient(90deg,#ea580c,#f59e0b)}
.bar.down{background:linear-gradient(90deg,#059669,#10b981)}
.delta{font-size:30px;font-weight:800;text-align:right}
.rulebox{padding:34px}
.rulebox h4{margin:0;font-size:25px;font-weight:800}
.rulebox .rl{margin-top:26px;display:grid;grid-template-columns:68px 1fr 78px;gap:16px;align-items:center;font-size:20px;font-weight:700;color:#d4d4d8}
.rulebox .track{height:16px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden}
.rulebox .fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#34d399,#10b981)}
.footnote{margin-top:44px;font-size:20px;color:#7c7c85;max-width:1180px;line-height:1.55}
.side{display:flex;gap:56px;margin-top:24px;align-items:flex-start}
.side .col{flex:1}
.side h3{font-size:26px;font-weight:800;letter-spacing:.02em;margin:0 0 26px;display:flex;align-items:center;gap:12px}
.side ul{list-style:none;display:flex;flex-direction:column;gap:26px}
.side li{display:flex;gap:16px;font-size:23px;line-height:1.4;color:#c4c4c9}
.side li b{color:#fafafa;font-weight:700}
.idx{flex:0 0 34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:800;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);color:#e4e4e7}
.idx.g{background:rgba(16,185,129,.16);border-color:rgba(16,185,129,.35);color:#34d399}
.idx.o{background:rgba(249,115,22,.16);border-color:rgba(249,115,22,.35);color:#fb923c}
"""

def footer(idx, total=5, brand=""):
    extra = f' &nbsp;·&nbsp; <b>{brand}</b>' if brand else ""
    return (f'<div class="footer"><span><b>LANDMARK QA/QC</b>{extra}'
            f' &nbsp;·&nbsp; ĐỘI 005 · WAR</span><span>{idx:02d} / {total:02d}</span></div>')

def shell(body):
    return ("<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            f"<style>{CSS}{EXTRA_CSS}</style></head><body><div class='slide'>"
            "<div class='grid'></div><div class='glow g1'></div><div class='glow g2'></div><div class='glow g3'></div>"
            f"{body}</div></body></html>")

# --------------------------------------------------------------------------- slide 1
s1 = f"""
<div class="title-wrap">
  <div class="title-left">
    <div class="kicker"><span class="bar"></span> VinFast Data Quality · Demo Day</div>
    <div class="big" style="margin-top:26px">LANDMARK<br/><span class="accent">QA/QC</span></div>
    <p class="lead">Kiểm định nhãn keypoint bằng <b>AI Ensemble</b> và <b>luật hình học</b>.<br/>
      Chỉ ra điểm nghi lỗi để reviewer quyết định nhanh — thay vì soi 100% bằng mắt.</p>
    <div class="team">
      <span class="pill">ĐỘI 005 · WAR</span>
      <span class="muted">Human Pose 17 &nbsp;·&nbsp; Face Landmark VF50</span>
    </div>
  </div>
  <div class="hero">
    {SKELETON}
    <div class="hud">
      <div class="k">Cảnh báo · frame 12 · kp 10</div>
      <div class="row"><span class="tag to">swap trái/phải</span><span class="v">0.92</span></div>
      <div class="row"><span class="tag tg">Temporal Q</span><span class="v">87%</span></div>
    </div>
  </div>
</div>
"""
slides = [shell(s1)]

# --------------------------------------------------------------------------- slide 2
def err_card(svg, title, desc):
    return f'<div class="card pcard"><div class="thumb">{svg}</div><h3>{title}</h3><p>{desc}</p></div>'

err_offset = """<svg viewBox="0 0 300 170">
  <rect x="92" y="22" width="116" height="102" rx="10" stroke="#3f3f46" stroke-width="2.5" fill="none"/>
  <g stroke="#52525b" stroke-width="2.5" stroke-linecap="round" fill="none">
    <circle cx="140" cy="46" r="8"/>
    <path d="M140 54v24"/><path d="M124 62h32"/>
    <path d="M140 78 129 104"/><path d="M140 78 151 104"/>
  </g>
  <circle cx="156" cy="62" r="7" fill="#a1a1aa"/>
  <circle cx="188" cy="88" r="7.5" fill="#f97316"/>
  <circle cx="188" cy="88" r="14" fill="none" stroke="#f97316" stroke-width="2" opacity="0.5"/>
  <path d="M156 62 L188 88" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5 6"/>
  <text x="150" y="148" fill="#8b8b93" font-size="15" font-family="Segoe UI" text-anchor="middle">Human vs AI · lệch Δ</text>
</svg>"""
err_swap = """<svg viewBox="0 0 300 170">
  <g stroke="#52525b" stroke-width="3" stroke-linecap="round" fill="none">
    <path d="M150 46v40"/><path d="M110 52h80"/>
    <path d="M110 52 96 108"/><path d="M190 52 204 108"/>
    <path d="M150 86 132 132"/><path d="M150 86 168 132"/>
  </g>
  <circle cx="96" cy="108" r="9" fill="#f97316"/>
  <circle cx="204" cy="108" r="9" fill="#f97316"/>
  <path d="M96 108 C 130 60, 170 60, 204 108" stroke="#ef4444" stroke-width="2.5" fill="none" stroke-dasharray="5 6"/>
  <text x="150" y="156" fill="#8b8b93" font-size="15" font-family="Segoe UI" text-anchor="middle">L / R bị hoán đổi</text>
</svg>"""
err_drift = """<svg viewBox="0 0 300 170">
  <g stroke="#3f3f46" stroke-width="2" fill="none">
    <rect x="30" y="44" width="66" height="82" rx="8"/>
    <rect x="117" y="44" width="66" height="82" rx="8"/>
    <rect x="204" y="44" width="66" height="82" rx="8"/>
  </g>
  <circle cx="63" cy="86" r="7" fill="#a1a1aa"/>
  <circle cx="150" cy="70" r="7" fill="#f97316"/>
  <circle cx="237" cy="98" r="7" fill="#f97316"/>
  <path d="M63 86 L150 70 L237 98" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5 6" fill="none"/>
  <text x="150" y="156" fill="#8b8b93" font-size="15" font-family="Segoe UI" text-anchor="middle">Frame t-1 · t · t+1</text>
</svg>"""

s2 = f"""
<div class="head">
  <div class="kicker"><span class="bar"></span> Bài toán</div>
  <h1 class="title">Nhãn thủ công sai ở đâu?</h1>
</div>
<div class="body">
<div class="cards3">
  {err_card(err_offset, "Lệch tọa độ", "Điểm vẽ không khớp vị trí người trong ảnh.")}
  {err_card(err_swap, "Hoán đổi trái/phải", "Swap tay, chân, mắt — model học sai hệ quy chiếu.")}
  {err_card(err_drift, "Trôi & giật theo frame", "Điểm nhảy bất thường giữa các frame liên tiếp.")}
</div>
<div class="notes">
  <span><span class="mk"></span>Kiểm 100% bằng mắt: chậm, thiếu nhất quán, không mở rộng được.</span>
  <span><span class="mk" style="background:#2563eb"></span>Nhãn bẩn → model học sai → tốn chi phí sửa về sau.</span>
</div>
</div>
{footer(2)}
"""
slides.append(shell(s2))

# --------------------------------------------------------------------------- slide 3
pipe_steps = [("ZIP / video", "dữ liệu vào"), ("AI inference", "Pose + Face"),
              ("Ghép &amp; đối chiếu", "annotation ↔ detection"), ("Chấm điểm", "nghi lỗi + xếp hạng"),
              ("Review", "từng điểm"), ("Export", "nhãn sạch + audit")]
pipe_html = ""
for i, (t, s) in enumerate(pipe_steps):
    if i:
        pipe_html += '<span class="arrow">→</span>'
    pipe_html += f'<div class="step">{t}<small>{s}</small></div>'

s3 = f"""
<div class="head">
  <div class="kicker"><span class="bar"></span> Giải pháp</div>
  <h1 class="title">Một pipeline, hai chuẩn nhãn</h1>
</div>
<div class="body">
<div class="pipe">{pipe_html}</div>
<div class="cards3">
  <div class="card fcard">
    <div class="cap"><span class="dot" style="background:#f97316"></span>Human Pose 17</div>
    <p>Ensemble YOLO26s + RTMPose hợp nhất thành vị trí đồng thuận, chấm điểm nghi ngờ theo chuẩn OKS.</p>
    <div class="kw"><span>Ensemble K2</span><span>OKS</span><span>Swap L/R</span></div>
  </div>
  <div class="card fcard">
    <div class="cap"><span class="dot" style="background:#38bdf8"></span>Face Landmark VF50</div>
    <p>MediaPipe 478 điểm ánh xạ về 50 điểm, kiểm định bằng 14 luật hình học và NME chuẩn hoá theo IOD.</p>
    <div class="kw"><span>14 luật R01–R14</span><span>NME</span><span>Hungarian</span></div>
  </div>
  <div class="card fcard">
    <div class="cap"><span class="dot" style="background:#34d399"></span>Temporal QA/QC</div>
    <p>Nối cùng một người qua nhiều frame, hồi quy quỹ đạo để gợi ý điểm và bắt nhãn nhảy bất thường.</p>
    <div class="kw"><span>Track xuyên frame</span><span>Cửa sổ 5–29</span><span>Gợi ý điểm</span></div>
  </div>
</div>
</div>
{footer(3)}
"""
slides.append(shell(s3))

# --------------------------------------------------------------------------- slide 4
def metric(name, sub, before, after, before_w, after_w, cls, delta, dcls):
    return f"""
    <div class="mrow">
      <div class="mname">{name}<small>{sub}</small></div>
      <div class="bars">
        <div class="bar k1" style="width:{before_w}%">K1 &nbsp;{before}</div>
        <div class="bar {cls}" style="width:{after_w}%">K2 &nbsp;{after}</div>
      </div>
      <div class="delta {dcls}">{delta}</div>
    </div>"""

s4 = f"""
<div class="head">
  <div class="kicker"><span class="bar"></span> Kết quả</div>
  <h1 class="title">Ensemble giảm báo động oan</h1>
  <p class="sub">Pose trên COCO TEST (500 ảnh, chia DEV/TEST, khoá ngưỡng trước khi đánh giá).</p>
</div>
<div class="body">
<div style="display:flex;gap:40px;margin-top:8px">
  <div class="chart">
    {metric("AUPRC", "chất lượng xếp hạng", "0.19", "0.28", 68, 100, "up", "+49%", "o")}
    {metric("Precision", "độ chính xác cảnh báo", "0.24", "0.39", 62, 100, "up", "+60%", "o")}
    {metric("False alarm", "tỉ lệ báo sai (thấp hơn tốt)", "5.8%", "2.7%", 100, 47, "down", "−53%", "g")}
  </div>
  <div class="card rulebox" style="flex:1">
    <h4>Face VF50 — F1 theo luật</h4>
    <div class="rl"><span>R01</span><div class="track"><div class="fill" style="width:100%"></div></div><span>1.00</span></div>
    <div class="rl"><span>R10</span><div class="track"><div class="fill" style="width:100%"></div></div><span>1.00</span></div>
    <div class="rl"><span>R14</span><div class="track"><div class="fill" style="width:100%"></div></div><span>1.00</span></div>
    <div class="rl"><span>R13</span><div class="track"><div class="fill" style="width:99%"></div></div><span>0.99</span></div>
    <div class="rl"><span>R11</span><div class="track"><div class="fill" style="width:94%"></div></div><span>0.94</span></div>
    <div class="rl"><span>R03</span><div class="track"><div class="fill" style="width:91%"></div></div><span>0.91</span></div>
  </div>
</div>
<p class="footnote">Dữ liệu công khai: COCO val2017, CrowdPose, WFLW, 300W, 300VW. Xu hướng K2 &gt; K1 giữ nguyên trên CrowdPose.
R13 trên video thật còn FPR 10.6% — đã ghi nhận là hạn chế, không tô hồng.</p>
</div>
{footer(4)}
"""
slides.append(shell(s4))

# --------------------------------------------------------------------------- slide 5
def li(i, text, cls="", tag="✓"):
    return f'<li><span class="idx {cls}">{tag}</span><span>{text}</span></li>'

s5 = f"""
<div class="head">
  <div class="kicker"><span class="bar"></span> Khả thi &amp; bước tiếp theo</div>
  <h1 class="title">Đã dùng được, còn dư địa mở rộng</h1>
</div>
<div class="body">
<div class="side">
  <div class="col">
    <h3><span class="dot" style="background:#34d399"></span>Hiện có</h3>
    <ul>
      {li(0, "<b>CPU-only, chạy offline</b> — không cần GPU hay cloud", "g")}
      {li(0, "<b>Nhận ảnh, ZIP và video</b> — trả kết quả theo từng frame", "g")}
      {li(0, "<b>Review từng điểm</b> — đổi nhãn, chấp nhận hoặc bỏ qua, có lịch sử", "g")}
      {li(0, "<b>Xuất nhãn sạch</b> — kèm log kiểm định để đối soát", "g")}
    </ul>
  </div>
  <div class="col">
    <h3><span class="dot" style="background:#fb923c"></span>Tiếp theo</h3>
    <ul>
      {li(1, "Tập <b>lỗi gán nhãn thật</b> làm TEST thực địa", "o", "1")}
      {li(2, "<b>Benchmark Temporal</b> có ground truth (điểm bị che)", "o", "2")}
      {li(3, "<b>Pilot downstream</b>: nhãn sạch giúp model học tốt hơn", "o", "3")}
      {li(4, "<b>Tinh chỉnh ngưỡng theo từng bộ dữ liệu</b> — ánh sáng, góc máy, mật độ người", "o", "4")}
    </ul>
  </div>
</div>
</div>
{footer(5)}
"""
slides.append(shell(s5))

# --------------------------------------------------------------------------- render
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
                    "--window-size=1920,1080", "--virtual-time-budget=3500",
                    f"--screenshot={op}", "file:///" + str(hp).replace("\\", "/")],
                   capture_output=True)
    print("rendered", op.name, op.stat().st_size)
    pngs.append(op)

imgs = [Image.open(p).convert("RGB") for p in pngs]
pdf = OUT / "LandmarkQA_QC_slides.pdf"
imgs[0].save(pdf, "PDF", resolution=144.0, save_all=True, append_images=imgs[1:])
print("PDF:", pdf, pdf.stat().st_size)
