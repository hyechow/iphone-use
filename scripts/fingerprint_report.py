"""生成语义指纹实验可视化报告（HTML）。"""

import base64
import json
import random
from pathlib import Path

import numpy as np

DATASET_DIR = Path("logs/dataset/微信")
PAGES_DIR = DATASET_DIR / "pages"
OUTPUT_HTML = DATASET_DIR / "fingerprint_report.html"

# ── 加载数据 ────────────────────────────────────────────────────────────────

results = json.load(open(DATASET_DIR / "semantic_exp_results.json"))
records = results["records"]
text_matrix = np.array(results["sim_matrix"])
n = len(records)

# 视觉相似度矩阵（重新计算）
import torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

print("Loading GUIClip...")
processor = CLIPProcessor.from_pretrained("Jl-wei/guiclip-vit-base-patch32", local_files_only=True)
clip_model = CLIPModel.from_pretrained("Jl-wei/guiclip-vit-base-patch32", local_files_only=True)
clip_model.eval()

vis_embs = []
for r in records:
    img = Image.open(PAGES_DIR / r["dir_name"] / "screenshot.png").convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        pixel = {k: v for k, v in inputs.items() if k.startswith("pixel")}
        emb = clip_model.visual_projection(clip_model.vision_model(**pixel).pooler_output)
        emb = F.normalize(emb, p=2, dim=1).squeeze(0).numpy()
    vis_embs.append(emb)
vis_matrix = np.array([[float(np.dot(vis_embs[i], vis_embs[j])) for j in range(n)] for i in range(n)])
print("GUIClip done.")

# ── 工具函数 ────────────────────────────────────────────────────────────────

def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def sim_color(s: float, good_thresh=0.90, bad_thresh=0.75) -> str:
    if s >= good_thresh: return "#22c55e"
    if s >= bad_thresh:  return "#f59e0b"
    return "#ef4444"

def cascade_decision(i: int, library: list[int], tv=0.80, tt=0.92):
    if not library:
        return "new", None, None
    vis_sims = [(vis_matrix[i][j], j) for j in library]
    vis_max, vis_nn = max(vis_sims)
    if vis_max > tv:
        return "skip_visual", vis_max, vis_nn
    txt_sims = [(text_matrix[i][j], j) for j in library]
    txt_max, txt_nn = max(txt_sims)
    if txt_max > tt:
        return "skip_text", txt_max, txt_nn
    return "new", max(vis_max, txt_max), None

# ── 仿真一次固定顺序（用于展示）────────────────────────────────────────────

random.seed(42)
order = list(range(n))
random.shuffle(order)
library: list[int] = []
decisions = []
for i in order:
    dec, score, nn = cascade_decision(i, library)
    decisions.append({"page_i": i, "decision": dec, "score": score, "nn_j": nn})
    if dec == "new":
        library.append(i)

# 最近邻错误案例
nn_errors = []
for i in range(n):
    sims = [(text_matrix[i][j], j) for j in range(n) if j != i]
    nn_sim, nn_j = max(sims)
    if records[i]["group"] != records[nn_j]["group"]:
        nn_errors.append((i, nn_j, nn_sim))
nn_errors.sort(key=lambda x: -x[2])

# ── HTML 生成 ────────────────────────────────────────────────────────────────

def card(title, content, color="#1e293b"):
    return f"""<div style="background:{color};border-radius:12px;padding:20px;margin:12px 0">{
        f'<div style="font-size:13px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">{title}</div>'
    }{content}</div>"""

def stat(label, value, sub=""):
    return f"""<div style="text-align:center;padding:12px 20px">
  <div style="font-size:28px;font-weight:700;color:#f1f5f9">{value}</div>
  <div style="font-size:12px;color:#94a3b8">{label}</div>
  {'<div style="font-size:11px;color:#64748b">'+sub+'</div>' if sub else ''}
</div>"""

def thumb(r, size=90, border_color="#334155"):
    path = PAGES_DIR / r["dir_name"] / "screenshot.png"
    b = img_b64(path)
    return f'<img src="data:image/png;base64,{b}" style="width:{size}px;border-radius:8px;border:2px solid {border_color};object-fit:cover">'

def fp_text(r):
    lines = r["fingerprint"].split("\n")[:3]
    return "<br>".join(f'<span style="color:#94a3b8">{l}</span>' for l in lines)

# ── Groups section ──────────────────────────────────────────────────────────

GROUPS = {}
for r in records:
    GROUPS.setdefault(r["group"], []).append(r)

groups_html = ""
for group, members in GROUPS.items():
    idxs = [records.index(r) for r in members]
    same_sims = [text_matrix[i][j] for i in idxs for j in idxs if i < j]
    vis_sims  = [vis_matrix[i][j]  for i in idxs for j in idxs if i < j]
    avg_t = np.mean(same_sims) if same_sims else 0
    avg_v = np.mean(vis_sims)  if vis_sims  else 0

    thumbs = "".join(
        f'<div style="text-align:center;margin:4px">'
        f'{thumb(r, 72)}'
        f'<div style="font-size:10px;color:#64748b;margin-top:4px">[{r["idx"]:03d}]</div>'
        f'</div>'
        for r in members
    )

    bar_t = f'<div style="background:#0f172a;border-radius:4px;height:8px;margin:4px 0"><div style="background:{sim_color(avg_t)};width:{avg_t*100:.0f}%;height:100%;border-radius:4px"></div></div>'
    bar_v = f'<div style="background:#0f172a;border-radius:4px;height:8px;margin:4px 0"><div style="background:{sim_color(avg_v, 0.80, 0.60)};width:{avg_v*100:.0f}%;height:100%;border-radius:4px"></div></div>'

    groups_html += f"""
<div style="background:#1e293b;border-radius:12px;padding:16px;margin:8px 0">
  <div style="display:flex;align-items:flex-start;gap:16px">
    <div style="flex:1">
      <div style="font-size:15px;font-weight:600;color:#f1f5f9;margin-bottom:8px">{group} <span style="color:#64748b;font-size:12px">× {len(members)}</span></div>
      <div style="font-size:11px;color:#64748b">文本同组相似度</div>
      {bar_t}
      <div style="font-size:11px;color:#f1f5f9;margin-bottom:6px">{avg_t:.3f}</div>
      <div style="font-size:11px;color:#64748b">视觉同组相似度</div>
      {bar_v}
      <div style="font-size:11px;color:#f1f5f9">{avg_v:.3f}</div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;max-width:520px">{thumbs}</div>
  </div>
</div>"""

# ── Errors section ──────────────────────────────────────────────────────────

errors_html = ""
for i, j, sim in nn_errors[:8]:
    ri, rj = records[i], records[j]
    vis_sim = vis_matrix[i][j]
    cascade = "视觉会命中" if vis_sim > 0.80 else ("语义会命中" if sim > 0.92 else "两关都不命中（正确处理）")
    cascade_color = "#ef4444" if vis_sim > 0.80 or sim > 0.92 else "#22c55e"

    errors_html += f"""
<div style="background:#1e293b;border-radius:12px;padding:16px;margin:8px 0">
  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="text-align:center">
      {thumb(ri, 80, "#6366f1")}
      <div style="font-size:11px;color:#6366f1;margin-top:4px">[{ri['idx']:03d}] {ri['group']}</div>
    </div>
    <div style="flex:1;padding-top:8px">
      <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">{fp_text(ri)}</div>
      <div style="margin:8px 0;display:flex;gap:12px">
        <span style="font-size:12px;color:#94a3b8">文本NN: <strong style="color:{sim_color(sim)}">{sim:.3f}</strong></span>
        <span style="font-size:12px;color:#94a3b8">视觉: <strong style="color:{sim_color(vis_sim,0.80,0.60)}">{vis_sim:.3f}</strong></span>
        <span style="font-size:12px;background:{cascade_color}22;color:{cascade_color};padding:2px 8px;border-radius:4px">{cascade}</span>
      </div>
      <div style="font-size:12px;color:#94a3b8;margin-top:6px">{fp_text(rj)}</div>
    </div>
    <div style="text-align:center">
      {thumb(rj, 80, "#ef4444")}
      <div style="font-size:11px;color:#ef4444;margin-top:4px">[{rj['idx']:03d}] {rj['group']}</div>
    </div>
  </div>
</div>"""

# ── Cascade decisions section ────────────────────────────────────────────────

cascade_html = ""
for d in decisions[:20]:
    i = d["page_i"]
    r = records[i]
    dec = d["decision"]
    score = d["score"]
    nn_j = d["nn_j"]

    if dec == "skip_visual":
        badge = f'<span style="background:#3b82f622;color:#3b82f6;padding:2px 8px;border-radius:4px;font-size:11px">视觉命中 {score:.3f}（省LLM）</span>'
        nn_r = records[nn_j]
        right = f'{thumb(nn_r, 60, "#3b82f6")}<div style="font-size:10px;color:#3b82f6;margin-top:3px">[{nn_r["idx"]:03d}]</div>'
    elif dec == "skip_text":
        badge = f'<span style="background:#8b5cf622;color:#8b5cf6;padding:2px 8px;border-radius:4px;font-size:11px">语义命中 {score:.3f}</span>'
        nn_r = records[nn_j]
        right = f'{thumb(nn_r, 60, "#8b5cf6")}<div style="font-size:10px;color:#8b5cf6;margin-top:3px">[{nn_r["idx"]:03d}]</div>'
    else:
        badge = f'<span style="background:#22c55e22;color:#22c55e;padding:2px 8px;border-radius:4px;font-size:11px">新页面 → 加入库</span>'
        right = ""

    correct = dec == "new" or (nn_j is not None and records[i]["group"] == records[nn_j]["group"])
    border = "#22c55e" if correct else "#ef4444"

    cascade_html += f"""
<div style="background:#1e293b;border-radius:10px;padding:12px;margin:6px 0;border-left:3px solid {border}">
  <div style="display:flex;gap:12px;align-items:center">
    {thumb(r, 60)}
    <div style="flex:1">
      <div style="font-size:12px;color:#f1f5f9;font-weight:600">[{r['idx']:03d}] {r['group']} · {r['name']}</div>
      <div style="font-size:11px;color:#64748b;margin:3px 0">{r['fingerprint'][:80]}...</div>
      {badge}
    </div>
    <div style="text-align:center">{right}</div>
  </div>
</div>"""

# ── Stats summary ────────────────────────────────────────────────────────────

same_sims = [text_matrix[i][j] for i in range(n) for j in range(i+1,n) if records[i]["group"]==records[j]["group"]]
diff_sims = [text_matrix[i][j] for i in range(n) for j in range(i+1,n) if records[i]["group"]!=records[j]["group"]]
vis_same  = [vis_matrix[i][j]  for i in range(n) for j in range(i+1,n) if records[i]["group"]==records[j]["group"]]
vis_diff  = [vis_matrix[i][j]  for i in range(n) for j in range(i+1,n) if records[i]["group"]!=records[j]["group"]]

skip_v = sum(1 for d in decisions if d["decision"]=="skip_visual")
skip_t = sum(1 for d in decisions if d["decision"]=="skip_text")
new_p  = sum(1 for d in decisions if d["decision"]=="new")

# ── 完整 HTML ────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>语义指纹实验报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #f1f5f9; font-family: -apple-system,system-ui,sans-serif; padding: 32px; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 600; color: #94a3b8; margin: 24px 0 10px; text-transform: uppercase; letter-spacing: .05em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 12px; }}
  .stats {{ display: flex; flex-wrap: wrap; background: #1e293b; border-radius: 12px; margin: 12px 0; }}
</style>
</head>
<body>

<h1>语义指纹实验报告</h1>
<div style="color:#64748b;font-size:13px;margin-bottom:24px">
  40 页面 × 8 组 | bge-small-zh + GUIClip | 级联方案 tv=0.80 tt=0.92
</div>

<h2>总体指标</h2>
<div class="stats">
  {stat("文本同组均值", f"{np.mean(same_sims):.3f}", f"min {np.min(same_sims):.3f}")}
  {stat("文本跨组均值", f"{np.mean(diff_sims):.3f}", f"max {np.max(diff_sims):.3f}")}
  {stat("文本间隔", f"{np.mean(same_sims)-np.mean(diff_sims):.3f}")}
  {stat("视觉同组均值", f"{np.mean(vis_same):.3f}", f"min {np.min(vis_same):.3f}")}
  {stat("视觉跨组均值", f"{np.mean(vis_diff):.3f}", f"max {np.max(vis_diff):.3f}")}
  {stat("视觉间隔", f"{np.mean(vis_same)-np.mean(vis_diff):.3f}")}
</div>

<h2>方案对比（F1@最佳阈值）</h2>
<div class="stats">
  {stat("text only", "0.864", "t=0.92")}
  {stat("visual only", "0.793", "t=0.80")}
  {stat("cascade", "0.876", "tv=0.85 tt=0.92")}
  {stat("视觉命中（省LLM）", f"{skip_v}", f"占 {skip_v/(skip_v+skip_t+new_p):.0%}")}
  {stat("语义命中", f"{skip_t}", f"占 {skip_t/(skip_v+skip_t+new_p):.0%}")}
  {stat("新页面", f"{new_p}", f"占 {new_p/(skip_v+skip_t+new_p):.0%}")}
</div>

<h2>各组页面一览</h2>
{groups_html}

<h2>文本最近邻错误案例（蓝=查询，红=错误NN）</h2>
{errors_html}

<h2>级联决策仿真（前20条，绿=正确，红=错误）</h2>
{cascade_html}

</body>
</html>"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"Report saved → {OUTPUT_HTML}")
