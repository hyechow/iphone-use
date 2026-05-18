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

from policy_expr.recon import PageParser, viz_result
from policy_expr.recon.bfs import probe_elements

ROOT = Path(__file__).parent.parent
LOG_ROOT = ROOT / "logs" / "recon"
KNOWLEDGE_ROOT = ROOT / "knowledge"
OFFLINE_EXPECTED_SIZE = (636, 1402)




# ── Core recon ────────────────────────────────────────────

def _parse_identity(phone) -> tuple:
    """Screenshot → parse page identity. Returns (png_bytes, knowledge, page_name)."""
    import re
    png_bytes = phone.screenshot()

    # print("\n解析页面身份...")
    knowledge = PageParser().analyze_screen(png_bytes)
    desc = knowledge.page.description[:20].strip()
    page_name = re.sub(r'[\\/:*?"<>|\s]+', '_', desc) or "page"
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

def run_app(app: str, depth: int = 0, sample: int = 0,
            mode: str | None = None, target: str | None = None) -> None:
    """Online recon with DFS depth control.

    depth=0: explore current page only.
    depth=N: DFS explore discovered pages (up to N levels deep).
    mode: None=initial, "add"=add page to existing app, "update"=re-probe target page.
    target: required for both "add" (parent page dir) and "update" (page dir to overwrite).
    """
    from policy_expr.perception import LivePhoneSession
    from policy_expr.recon.dfs import explore_dfs

    if mode in ("add", "update") and not target:
        label = "父页面目录名" if mode == "add" else "页面目录名"
        print(f"错误: --mode {mode} 需要指定 --target <{label}>")
        raise SystemExit(1)

    mode_label = {"add": "新增", "update": "更新"}.get(mode, "侦察")
    print(f"应用{mode_label}: {app} (depth={depth}, sample={sample})")
    app_log_dir = LOG_ROOT / app
    app_log_dir.mkdir(parents=True, exist_ok=True)

    from policy_expr.runner import _tee_stdio
    with _tee_stdio(app_log_dir):
        # Preload GUIClip model to avoid loading it during exploration
        print("预加载 GUIClip 模型...")
        from policy_expr.recon.back_nav import _get_identity_comp
        import io
        from PIL import Image

        # Create a dummy 1x1 white image to trigger model loading
        dummy_img = io.BytesIO()
        Image.new('RGB', (1, 1), color='white').save(dummy_img, format='PNG')
        dummy_bytes = dummy_img.getvalue()

        # Call similarity once to trigger model loading
        comp = _get_identity_comp()
        comp.raw_similarity(dummy_bytes, dummy_bytes)

        with LivePhoneSession() as phone:
            # Phase 1: DFS exploration (probe only, no knowledge gen)
            tree = explore_dfs(phone, app_log_dir, max_depth=depth, sample=sample,
                               mode=mode, target_dir=target)

        # Phase 2: Post-order knowledge generation (leaves first) — DISABLED
        # total = sum(1 + _count_tree_nodes(n.children) for n in tree)
        # if total > 0:
        #     print(f"\n--- 开始知识生成 (自底向上, {total} 个页面) ---")
        #     generate_knowledge_postorder(tree, app)
        #     print("知识生成完成")


def _count_tree_nodes(nodes) -> int:
    from policy_expr.recon.dfs import _count_nodes
    return _count_nodes(nodes)


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


def run_export(app: str, page: str | None = None) -> None:
    """Export page knowledge for all (or one specific) pages of an app.

    Reads initial_result.json + recon_result.json per page, runs one LLM call,
    writes page_meta.json + knowledge.md locally and syncs to knowledge/{app}/.
    """
    from policy_expr.self_learning.knowledge import build_export, save_export

    app_log_dir = LOG_ROOT / app
    if page:
        page_dirs = [app_log_dir / page]
    else:
        page_dirs = sorted(
            p for p in app_log_dir.iterdir()
            if p.is_dir() and (p / "recon_result.json").exists()
        )

    print(f"\n--- 导出页面知识: {app} ({len(page_dirs)} 个页面) ---")
    for page_dir in page_dirs:
        print(f"\n  [{page_dir.name}]")
        try:
            exported = build_export(page_dir)
            save_export(exported, page_dir, KNOWLEDGE_ROOT / app)
            print(f"  ✓ {exported.meta.page_title} ({exported.meta.page_type})")
        except Exception as e:
            print(f"  ✗ {e}")


# ── CLI ──────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="App Recon: 页面结构分析与自学习")
    ap.add_argument("--app", type=str, metavar="NAME", help="在线侦察指定应用，自动完成 recon→learn→knowledge")
    ap.add_argument("--depth", type=int, default=0, metavar="N", help="DFS 探索深度，0=仅当前页面 (默认)")
    ap.add_argument("--sample", type=int, default=0, metavar="N", help="每个页面随机采样 N 个元素探测，0=全部 (默认)")
    ap.add_argument("--mode", choices=["add", "update"], metavar="MODE", help="add=新增页面到已有应用, update=重新探测指定页面")
    ap.add_argument("--target", type=str, metavar="DIR", help="add=父页面目录名, update=要更新的页面目录名")
    ap.add_argument("--export", type=str, metavar="APP", help="导出指定应用的页面知识（page_meta.json + knowledge.md）")
    ap.add_argument("--page", type=str, metavar="DIR", help="--export 时只导出指定页面目录")
    ap.add_argument("--debug-parse", nargs="*", type=Path, metavar="PATH", help="[调试] 解析图片，无参数则在线截图")
    args = ap.parse_args()

    if args.app:
        run_app(args.app, depth=args.depth, sample=args.sample,
                mode=args.mode, target=args.target)
    elif args.export:
        run_export(args.export, page=args.page)
    elif args.debug_parse is not None:
        run_parse(args.debug_parse)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
