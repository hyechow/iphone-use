"""CLI entry point for app reconnaissance and self-learning."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from policy_expr.executor import logical_xy
from policy_expr.recon import PageParser, viz_result
from policy_expr.recon.bfs import probe_elements
from policy_expr.recon.back_nav import return_to_initial
from policy_expr.recon.utils import ProbeAbortedError
from policy_expr.trace import BfsTracer

ROOT = Path(__file__).parent.parent
LOG_ROOT = ROOT / "logs" / "recon"
KNOWLEDGE_ROOT = ROOT / "knowledge"
OFFLINE_EXPECTED_SIZE = (636, 1402)


# ── Helpers ──────────────────────────────────────────────

def _parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end < 0:
        return {}
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _check_knowledge_exists(app: str, signature: str) -> Path | None:
    app_dir = KNOWLEDGE_ROOT / app
    if not app_dir.is_dir():
        return None
    for md_path in sorted(app_dir.glob("*.md")):
        if _parse_frontmatter(md_path).get("signature") == signature:
            return md_path
    return None


def _resolve_knowledge_path(recon_dir: Path) -> Path:
    """logs/recon/{app}/{page} → knowledge/{app}/{page}.md"""
    try:
        parts = recon_dir.resolve().relative_to(LOG_ROOT.resolve()).parts
        app = parts[0] if len(parts) >= 2 else "unknown"
        page = parts[1] if len(parts) >= 2 else parts[0]
    except (ValueError, IndexError):
        app, page = "unknown", recon_dir.name
    return KNOWLEDGE_ROOT / app / f"{page}.md"


def _derive_page_name(signature: str) -> str:
    """Extract stable page name from signature's 2nd segment."""
    parts = signature.split("/")
    return parts[1] if len(parts) >= 2 else parts[0]


# ── Core recon ────────────────────────────────────────────

def _parse_identity(phone) -> tuple:
    """Screenshot → parse page identity. Returns (png_bytes, knowledge, page_name)."""
    png_bytes = phone.screenshot()

    # print("\n解析页面身份...")
    knowledge = PageParser().analyze_screen(png_bytes)
    page_name = _derive_page_name(knowledge.page.identity.signature)
    # print(f"  页面: {page_name}  签名: {knowledge.page.identity.signature}")
    return png_bytes, knowledge, page_name


def _probe_page(phone, knowledge, png_bytes, out_dir: Path, sample: int = 0,
                nav_stack: list | None = None):
    """Save initial screenshot + probe all elements. Returns (result, out_dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "initial.png"
    img_path.write_bytes(png_bytes)
    viz_result(knowledge, png_bytes, "initial", out_dir)

    result = probe_elements(phone.client, knowledge, out_dir, img_path, phone.screenshot,
                            sample=sample, nav_stack=nav_stack)
    result.save(out_dir / "recon_result.json")

    ok = sum(1 for t in result.taps if t.tap_ok and t.navigated)
    print(f"  探测完成: {len(result.taps)} 个元素 (成功导航 {ok})")

    return result, out_dir


# ── Commands ─────────────────────────────────────────────

def run_app(app: str, depth: int = 0, sample: int = 0) -> None:
    """Online recon with BFS depth control.

    depth=0: explore current page only.
    depth=N: BFS explore discovered target pages (up to N levels deep).

    Queue stores navigation chains: each item is (depth, nav_chain) where
    nav_chain is a list of (lx, ly, label) taps from root to target page.
    """
    import time
    from collections import deque

    from policy_expr.perception import LivePhoneSession

    print(f"应用侦察: {app} (depth={depth}, sample={sample})")

    app_log_dir = LOG_ROOT / app

    with LivePhoneSession() as phone:
        # nav_chain: taps from root to target, [] = root page
        queue: deque[tuple[int, list]] = deque([(depth, [])])
        root_ctx = None  # (page, initial_bytes, out_dir) for return_to_initial

        tracer = BfsTracer()
        trace_path = app_log_dir / "bfs_trace.json"
        chain_to_page: dict[tuple, str] = {}  # nav_chain_tuple → page_name
        visited_pages: set[str] = set()       # page names seen in this BFS run

        while queue:
            cur_depth, nav_chain = queue.popleft()

            # Return to root then navigate to target
            nav_stack: list[tuple[bytes, tuple[float, float] | None]] = []
            if root_ctx and nav_chain:
                root_page, root_bytes, root_out = root_ctx
                ok = return_to_initial(
                    phone.client, phone.screenshot,
                    [(root_bytes, nav_chain[0][:2] if nav_chain else None)],
                )[0]
                if not ok:
                    print(f"\n  \033[31m✗ 返回根页面失败，跳过: {[l for _, _, l in nav_chain]}\033[0m")
                    continue
                # Build nav_stack during navigation
                nav_stack = [(root_bytes, nav_chain[0][:2] if len(nav_chain) > 0 else None)]
                root_title = root_page.identity.page_title
                path_desc = " → ".join([root_title] + [l for _, _, l in nav_chain])
                print(f"\n\033[36m━━━ 页面跳转: {path_desc} ━━━\033[0m")
                for step_i, (lx, ly, label) in enumerate(nav_chain):
                    phone.client.tap(lx, ly)
                    time.sleep(2.0)
                    page_bytes = phone.screenshot()
                    next_coords = nav_chain[step_i + 1][:2] if step_i + 1 < len(nav_chain) else None
                    nav_stack.append((page_bytes, next_coords))

            # Explore current page
            png_bytes, knowledge, page_name = _parse_identity(phone)
            signature = knowledge.page.identity.signature

            # Derive parent info from nav_chain before any early-continue
            chain_key = tuple((lx, ly, label) for lx, ly, label in nav_chain)
            parent_page = chain_to_page.get(chain_key[:-1]) if chain_key else None
            via_tap = nav_chain[-1][2] if nav_chain else None
            chain_to_page[chain_key] = page_name

            # Only record in trace once per page — re-visits (e.g. BFS cycles back
            # to root via a different path) must not overwrite the original entry or
            # add a spurious child relationship that breaks report rendering.
            if page_name not in visited_pages:
                visited_pages.add(page_name)
                tracer.record_page(page_name, parent_page, via_tap, len(nav_chain))
                tracer.save(trace_path)
            else:
                print(f"  已访问，跳过记录: {page_name}")
                continue

            already_learned = _check_knowledge_exists(app, signature)
            if already_learned and cur_depth <= 0:
                print(f"  已学习过，跳过: {already_learned}")
                continue

            # Build nav_stack for root page (no nav_chain)
            if not nav_stack:
                nav_stack = [(png_bytes, None)]

            try:
                result, out_dir = _probe_page(phone, knowledge, png_bytes, app_log_dir / page_name,
                                              sample=sample, nav_stack=nav_stack)
            except ProbeAbortedError as e:
                tracer.record_error(page_name, e)
                tracer.save(trace_path)
                raise
            except Exception as e:
                tracer.record_error(page_name, e)
                tracer.save(trace_path)
                raise

            # TODO: For deeper BFS, we may want to learn/knowledge after each page to improve return_to_initial success. But it slows down the process and may cause more API costs, so we can enable it later if needed.
            if not already_learned:
                run_learn(out_dir / "recon_result.json")
                run_knowledge(out_dir)

            if root_ctx is None:
                root_ctx = (knowledge.page, (out_dir / "initial.png").read_bytes(), out_dir)

            if cur_depth <= 0:
                continue

            n_children = sum(1 for t in result.taps if t.tap_ok and t.navigated)
            if n_children:
                input(f"  发现 {n_children} 个可导航子页面，按回车继续探索...")

            for tap in result.taps:
                if not tap.tap_ok or not tap.navigated:
                    continue

                lx, ly = logical_xy(tap.x, tap.y)
                queue.append((cur_depth - 1, nav_chain + [(lx, ly, tap.label)]))


def run_parse(paths: list[Path]) -> None:
    """Parse page structure. With paths: offline. Without: online."""
    if paths:
        _parse_offline(paths)
    else:
        _parse_online()


def _parse_online() -> None:
    from policy_expr.perception import LivePhoneSession

    print("在线解析: 截取手机屏幕...")
    with LivePhoneSession() as phone:
        png_bytes, knowledge, page_name = _parse_identity(phone)
        _probe_page(phone, knowledge, png_bytes, LOG_ROOT / "online" / page_name)


def _parse_offline(paths: list[Path]) -> None:
    images: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".png":
            images.append(p)
        elif p.is_dir():
            images.extend(sorted(p.glob("*.png")))
    if not images:
        print("未找到 PNG 图片")
        return

    print(f"离线解析: 共 {len(images)} 张图片")
    parser = PageParser()
    out_dir = LOG_ROOT / "offline"
    for img_path in images:
        with Image.open(img_path) as img:
            if img.size != OFFLINE_EXPECTED_SIZE:
                print(f"截图尺寸不匹配: {img_path} 实际 {img.size}，期望 {OFFLINE_EXPECTED_SIZE}")
                raise SystemExit(1)
        png_bytes = img_path.read_bytes()
        initial_path = out_dir / f"{img_path.stem}_initial.png"
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_bytes(png_bytes)
        print(f"\n图片: {img_path.name}")
        knowledge = parser.analyze_screen(png_bytes)
        viz_result(knowledge, png_bytes, img_path.stem, out_dir)


def run_learn(recon_path: Path) -> None:
    """Build page flow descriptions from a recon result."""
    from policy_expr.self_learning.flow import build_page_flows, save_page_flows

    print(f"\n--- 生成功能描述 ---")
    page_flow = build_page_flows(recon_path)
    out_path = recon_path.parent / "page_flows.json"
    save_page_flows(page_flow, out_path)

    print(f"功能流程: {len(page_flow.flows)} 条")
    for f in page_flow.flows:
        print(f"  {f.flow_description}")
    print(f"结果: {out_path}")


def run_knowledge(recon_dir: Path) -> None:
    """Build page operation knowledge skill from recon results."""
    from policy_expr.self_learning.knowledge import build_knowledge, save_knowledge

    print(f"\n--- 构建知识库 ---")
    kb = build_knowledge(recon_dir)

    save_knowledge(kb, recon_dir / "knowledge.md")
    out_path = _resolve_knowledge_path(recon_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_knowledge(kb, out_path)

    print(kb.to_skill())
    print(f"\n知识库: {out_path}")


# ── CLI ──────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="App Recon: 页面结构分析与自学习")
    ap.add_argument("--app", type=str, metavar="NAME", help="在线侦察指定应用，自动完成 recon→learn→knowledge")
    ap.add_argument("--depth", type=int, default=0, metavar="N", help="BFS 探索深度，0=仅当前页面 (默认)")
    ap.add_argument("--sample", type=int, default=0, metavar="N", help="每个页面随机采样 N 个元素探测，0=全部 (默认)")
    ap.add_argument("--debug-parse", nargs="*", type=Path, metavar="PATH", help="[调试] 解析图片，无参数则在线截图")
    ap.add_argument("--debug-learn", type=Path, metavar="JSON", help="[调试] 从侦察结果生成功能流程描述")
    ap.add_argument("--debug-knowledge", type=Path, metavar="DIR", help="[调试] 从侦察目录构建页面操作知识库")
    args = ap.parse_args()

    if args.app:
        run_app(args.app, depth=args.depth, sample=args.sample)
    elif args.debug_learn:
        run_learn(args.debug_learn)
    elif args.debug_knowledge:
        run_knowledge(args.debug_knowledge)
    elif args.debug_parse is not None:
        run_parse(args.debug_parse)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
