"""Generate a self-contained HTML tree visualization of the page transition graph.

No external dependencies. Screenshots referenced via relative paths (same as report_builder.py).
Each node shows the page's initial.png thumbnail; edges show tap labels.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_transition_graph(trace_path: Path, output_path: Path) -> None:
    """Read trace.json and write a self-contained HTML transition tree."""
    if not trace_path.exists():
        return

    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        transitions, pages_list = [], raw
    else:
        transitions, pages_list = raw.get("transitions", []), raw.get("pages", [])

    if not transitions:
        return

    app_log_dir = trace_path.parent

    # --- build adjacency list -----------------------------------------------

    # {src: [{tap, dst, status}]}
    children: dict[str, list[dict]] = {}
    for t in transitions:
        src = t["src"]
        if not src:
            continue
        children.setdefault(src, []).append(
            {"tap": t["tap"], "dst": t["dst"], "status": t["status"]}
        )

    # root = depth-0 page from trace pages list, fallback to node with no parent
    root_page = next((p["page"] for p in pages_list if p.get("depth") == 0), None)
    if not root_page:
        all_dst = {t["dst"] for t in transitions}
        root_page = next((src for src in children if src not in all_dst), None)
    if not root_page:
        return

    # --- screenshot helper ---------------------------------------------------

    def screenshot_rel(page_name: str, parent_name: str = "", tap_label: str = "") -> str:
        # Prefer the page's own initial.png (exists when the page was probed)
        img = app_log_dir / page_name / "initial.png"
        if img.exists():
            return f"./{page_name}/initial.png"
        # Fallback: the tap screenshot saved in the parent's tap directory
        if parent_name and tap_label:
            tap_dir = app_log_dir / parent_name / "tap"
            if tap_dir.exists():
                for f in sorted(tap_dir.glob(f"tap_*_{tap_label}.png")):
                    return f"./{parent_name}/tap/{f.name}"
        return ""

    # --- recursive HTML render ----------------------------------------------

    rendered: set[str] = set()

    def render_node(page_name: str, tap_label: str | None, status: str, parent_name: str = "") -> str:
        img_src = screenshot_rel(page_name, parent_name, tap_label or "")
        img_html = (
            f'<img src="{img_src}" class="thumb">'
            if img_src
            else '<div class="thumb thumb-missing"></div>'
        )

        node_cls = {
            "depth_limit": "node node-depth",
            "skipped_visited": "node node-skipped",
            "skipped_known": "node node-skipped",
        }.get(status, "node")

        badge = {
            "depth_limit": '<span class="badge badge-depth">未探测</span>',
            "skipped_visited": '<span class="badge badge-skip">已访问</span>',
            "skipped_known": '<span class="badge badge-skip">已知</span>',
        }.get(status, "")

        tap_html = (
            f'<div class="tap-row">'
            f'<span class="tap-connector"></span>'
            f'<span class="tap-label">{tap_label}</span>'
            f'</div>'
            if tap_label
            else ""
        )

        # Expand children only once per page (guard against revisit cycles)
        already = page_name in rendered
        rendered.add(page_name)

        kids_html = ""
        if not already and status not in ("skipped_visited", "skipped_known", "depth_limit"):
            for child in children.get(page_name, []):
                kids_html += render_node(child["dst"], child["tap"], child["status"], parent_name=page_name)

        children_div = f'<div class="children">{kids_html}</div>' if kids_html else ""

        return (
            f'<div class="node-group">'
            f'{tap_html}'
            f'<div class="{node_cls}">'
            f'{img_html}'
            f'<div class="node-label">{page_name}{badge}</div>'
            f'</div>'
            f'{children_div}'
            f'</div>'
        )

    body_html = render_node(root_page, None, "entered")
    title = app_log_dir.name

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>页面跳转图 — {title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, "PingFang SC", sans-serif;
      background: #f1f5f9; padding: 24px; color: #1e293b;
    }}
    h2 {{ font-size: 14px; color: #64748b; margin-bottom: 20px; font-weight: 500; }}

    /* ── tree structure ── */
    .node-group {{
      display: flex; flex-direction: column; align-items: flex-start;
    }}
    .children {{
      margin-left: 52px;
      padding-left: 20px;
      border-left: 2px solid #cbd5e1;
      display: flex; flex-direction: column; gap: 12px;
      margin-top: 0; padding-top: 0;
    }}
    .children > .node-group {{ margin-top: 0; }}

    /* ── tap arrow label ── */
    .tap-row {{
      display: flex; align-items: center; gap: 6px;
      margin: 6px 0 6px 0; padding-left: 0;
    }}
    .tap-connector {{
      display: inline-block; width: 20px; height: 2px;
      background: #94a3b8; flex-shrink: 0;
    }}
    .tap-label {{
      font-size: 11px; color: #475569;
      background: #e2e8f0; border-radius: 8px;
      padding: 2px 8px; max-width: 180px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}

    /* ── page node ── */
    .node {{
      display: flex; align-items: flex-end; gap: 8px;
      background: white; border: 1.5px solid #e2e8f0;
      border-radius: 10px; padding: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      cursor: default;
    }}
    .node:hover {{ border-color: #93c5fd; }}
    .node-depth {{
      background: #f8fafc; border-color: #cbd5e1;
      opacity: 0.7;
    }}
    .node-skipped {{
      background: #fffbeb; border: 1.5px dashed #d4a;
      opacity: 0.75;
    }}

    /* ── screenshot thumb ── */
    .thumb {{
      width: 54px; height: 96px; object-fit: cover;
      border-radius: 6px; border: 1px solid #e2e8f0;
      flex-shrink: 0; display: block;
    }}
    .thumb-missing {{
      background: #f1f5f9;
    }}

    /* ── label & badge ── */
    .node-label {{
      font-size: 12px; font-weight: 500; color: #334155;
      max-width: 110px; word-break: break-all; line-height: 1.4;
      display: flex; flex-direction: column; gap: 4px; align-self: center;
    }}
    .badge {{
      display: inline-block; font-size: 10px; padding: 1px 6px;
      border-radius: 8px; font-weight: 400;
    }}
    .badge-depth {{ background: #f1f5f9; color: #94a3b8; border: 1px solid #cbd5e1; }}
    .badge-skip  {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }}

    /* ── legend ── */
    .legend {{
      display: flex; gap: 20px; font-size: 11px; color: #94a3b8;
      margin-bottom: 20px; flex-wrap: wrap;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-box {{
      width: 20px; height: 34px; border-radius: 4px; border: 1.5px solid;
    }}
  </style>
</head>
<body>
  <h2>页面跳转图 — {title}</h2>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-box" style="background:white;border-color:#e2e8f0"></div>已探测
    </div>
    <div class="legend-item">
      <div class="legend-box" style="background:#f8fafc;border-color:#cbd5e1;opacity:.7"></div>仅记录（depth边界）
    </div>
    <div class="legend-item">
      <div class="legend-box" style="background:#fffbeb;border:1.5px dashed #d4a"></div>跳过（已访问/已知）
    </div>
  </div>
  {body_html}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  跳转图已保存: {output_path}")
