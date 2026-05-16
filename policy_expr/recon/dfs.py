"""DFS page exploration: depth-first traversal with post-order knowledge generation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from policy_expr.executor import logical_xy
from policy_expr.perception import try_resume_mac
from policy_expr.recon.back_nav import manual_recover as _manual_recover
from policy_expr.recon.bfs import probe_elements
from policy_expr.recon.back_nav import return_to_initial
from policy_expr.recon.utils import ProbeAbortedError
from policy_expr.trace import Tracer


@dataclass
class DfsPageNode:
    """One page in the DFS tree. Carries all state needed for deferred knowledge gen."""
    page_name: str
    signature: str
    parent: str | None
    via_tap: str | None
    depth: int
    nav_chain: list[tuple[float, float, str]]  # (lx, ly, label) from root
    out_dir: Path
    recon_result: object | None = None  # ReconResult, lazy import to avoid cycle
    children: list[DfsPageNode] = field(default_factory=list)
    error: dict | None = None
    knowledge_generated: bool = False


def explore_dfs(phone, app_log_dir: Path, max_depth: int = 0,
                sample: int = 0, similarity_threshold: float = 0.5) -> list[DfsPageNode]:
    """Top-level DFS exploration.

    Phone must be on the root page.
    Returns tree of explored pages (children attached).
    """
    from policy_expr.recon.page_parser import PageParser
    from policy_expr.recon_cli import _derive_page_name
    from policy_expr.recon.page_compare import make_comparator

    app = app_log_dir.name
    tracer = Tracer()
    trace_path = app_log_dir / "trace.json"
    visited_screenshots: list[bytes] = []
    visited_page_entries: list[tuple[str, bytes]] = []
    from policy_expr.recon.back_nav import _get_identity_comp
    id_comp = _get_identity_comp()
    chain_to_page: dict[tuple, str] = {}

    # Parse root page identity
    png_bytes = phone.screenshot()
    knowledge = PageParser().analyze_screen(png_bytes)
    page_name = _derive_page_name(knowledge.page.identity.signature)
    signature = knowledge.page.identity.signature

    nav_stack: list[tuple[bytes, tuple[float, float] | None]] = [(png_bytes, None)]

    # Record root
    visited_screenshots.append(png_bytes)
    visited_page_entries.append((page_name, png_bytes))
    tracer.record_page(page_name, None, None, 0)
    tracer.save(trace_path)

    # Map root's empty chain to page name so children can find their parent
    chain_to_page[()] = page_name

    # Check if root already learned
    already_learned = _check_knowledge_exists(app, signature)
    if already_learned and max_depth <= 0:
        print(f"  已学习过，跳过: {already_learned}")
        return []

    root_node = DfsPageNode(
        page_name=page_name, signature=signature,
        parent=None, via_tap=None, depth=0,
        nav_chain=[], out_dir=app_log_dir / page_name,
    )

    root_ctx = (knowledge.page, png_bytes, app_log_dir / page_name)

    # Probe root page with DFS callback
    def _on_root_element_tapped(area, after_bytes, navigated, tap_index, tap_result):
        """Callback for root page exploration. Enter child pages if depth allows."""
        print(f"    [DEBUG ROOT] _on_root_element_tapped called: area={area.label}, navigated={navigated}, max_depth={max_depth}")
        if not navigated or max_depth <= 0:
            print(f"    [DEBUG ROOT] Skipping child entry (navigated={navigated}, max_depth={max_depth})")
            return True  # Continue probing

        # Enter child page
        print(f"\n  → 进入子页面「{area.label}」")
        print(f"    [DEBUG ROOT] Entering child page: max_depth={max_depth}, will pass {max_depth - 1} to child")
        ax, ay = area.center_xy
        lx, ly = logical_xy(ax, ay)
        child_chain = [(lx, ly, area.label)]

        # Build child nav_stack with parent forward_coords
        parent_bytes, _ = nav_stack[-1]
        child_nav_stack = nav_stack[:-1] + [(parent_bytes, (lx, ly)), (after_bytes, None)]

        # Recursive DFS into child
        child_node = _dfs_recursive(
            phone, child_chain, child_nav_stack, root_ctx,
            chain_to_page, visited_screenshots, visited_page_entries, id_comp, tracer, trace_path,
            app_log_dir, max_depth - 1, sample,
        )

        if child_node is not None:
            root_node.children.append(child_node)

        # Return to root page
        print(f"\n  ← 返回「{page_name}」")
        ok, back_log = return_to_initial(
            phone.client, phone.screenshot, nav_stack,
            before_back_bytes=phone.screenshot(),
        )
        if not ok:
            recovered = _manual_recover(
                phone.client, phone.screenshot, nav_stack,
                len(nav_stack) - 1,
                prompt=f"子页面探索后无法返回 {page_name}",
            )
            if not recovered:
                raise ProbeAbortedError(
                    f"DFS: 无法从子页面返回 {page_name}",
                    failed_tap=-1, failed_element="",
                    back_attempts=back_log,
                )

        # Update tap result with back attempts
        tap_result.back_attempts = back_log

        return True  # Continue probing next element

    try:
        root_node.recon_result, root_node.out_dir = _probe_page_dfs(
            phone, knowledge, png_bytes, root_node.out_dir,
            sample=sample, nav_stack=nav_stack,
            on_element_tapped=_on_root_element_tapped,
        )
    except ProbeAbortedError as e:
        tracer.record_error(page_name, e)
        tracer.save(trace_path)
        raise

    # DFS into children is now handled inside _probe_page_dfs via callback

    # Post-exploration: pairwise similarity report
    if len(visited_page_entries) > 1:
        compute_pairwise_similarity(
            visited_page_entries,
            threshold=similarity_threshold,
            save_path=app_log_dir / "similarity_report.json",
        )

    # Post-exploration: transition graph
    from policy_expr.recon.viz_transitions import generate_transition_graph
    generate_transition_graph(trace_path, app_log_dir / "transition_graph.html")

    return [root_node]


def _dfs_explore_children(
    phone,
    node: DfsPageNode,
    nav_stack: list[tuple[bytes, tuple[float, float] | None]],
    root_ctx: tuple,
    chain_to_page: dict[tuple, str],
    visited_screenshots: list[bytes],
    id_comp,
    tracer: Tracer,
    trace_path: Path,
    app_log_dir: Path,
    remaining_depth: int,
    sample: int,
) -> None:
    """DFS into navigated children of node. Phone is on node's page."""
    from policy_expr.recon_cli import _check_knowledge_exists

    app = app_log_dir.name
    navigated_taps = [t for t in node.recon_result.taps if t.tap_ok and t.navigated]
    if not navigated_taps:
        return

    print(f"\n  发现 {len(navigated_taps)} 个可导航子页面")

    for tap in navigated_taps:
        lx, ly = logical_xy(tap.x, tap.y)
        child_chain = node.nav_chain + [(lx, ly, tap.label)]

        # 1. Tap into child page
        print(f"\n  → 进入子页面「{tap.label}」")
        phone.client.tap(lx, ly)
        time.sleep(2.0)
        child_bytes = phone.screenshot()

        # 2. Build child nav_stack with parent forward_coords
        # Parent entry's forward_coords points to the tap that re-enters this child
        parent_bytes, _ = nav_stack[-1]
        child_nav_stack = nav_stack[:-1] + [(parent_bytes, (lx, ly)), (child_bytes, None)]

        # 3. Recursive DFS into child
        child_node = _dfs_recursive(
            phone, child_chain, child_nav_stack, root_ctx,
            chain_to_page, visited_screenshots, [],
            id_comp, tracer, trace_path,
            app_log_dir, remaining_depth - 1, sample,
        )

        if child_node is not None:
            node.children.append(child_node)

        # 4. Back to this page (one level up)
        print(f"\n  ← 返回「{node.page_name}」")
        ok, back_log = return_to_initial(
            phone.client, phone.screenshot, nav_stack,
            before_back_bytes=phone.screenshot(),
        )
        if not ok:
            recovered = _manual_recover(
                phone.client, phone.screenshot, nav_stack,
                len(nav_stack) - 1,
                prompt=f"子页面探索后无法返回 {node.page_name}",
            )
            if not recovered:
                raise ProbeAbortedError(
                    f"DFS: 无法从子页面返回 {node.page_name}",
                    failed_tap=-1, failed_element="",
                    back_attempts=back_log,
                )


def _dfs_recursive(
    phone,
    nav_chain: list[tuple[float, float, str]],
    nav_stack: list[tuple[bytes, tuple[float, float] | None]],
    root_ctx: tuple,
    chain_to_page: dict[tuple, str],
    visited_screenshots: list[bytes],
    visited_page_entries: list[tuple[str, bytes]],
    id_comp,
    tracer: Tracer,
    trace_path: Path,
    app_log_dir: Path,
    max_depth: int,
    sample: int,
) -> DfsPageNode | None:
    """Recursive DFS step. Phone is on the target page.

    PRE:  phone is on this page (nav_stack top)
    POST: phone is back on this page (after all children explored)
    """
    from policy_expr.recon.page_parser import PageParser
    from policy_expr.recon_cli import _check_knowledge_exists, _derive_page_name

    app = app_log_dir.name

    # Parse page identity
    png_bytes, knowledge, page_name = _parse_identity(phone)
    signature = knowledge.page.identity.signature

    # Trace + visited check (using GUIClip for visual page identity)
    chain_key = tuple(nav_chain)
    parent_page = chain_to_page.get(chain_key[:-1]) if chain_key else None
    via_tap = nav_chain[-1][2] if nav_chain else None
    chain_to_page[chain_key] = page_name

    _src = parent_page or ""
    _tap = via_tap or ""

    # Check if this page has been visited before (by visual similarity)
    for visited_bytes in visited_screenshots:
        if id_comp.is_same_page(visited_bytes, png_bytes).matched:
            print(f"  已访问（视觉相似），跳过: {page_name}")
            tracer.record_transition(_src, _tap, page_name, "skipped_visited")
            tracer.save(trace_path)
            return None

    visited_screenshots.append(png_bytes)
    visited_page_entries.append((page_name, png_bytes))
    tracer.record_page(page_name, parent_page, via_tap, len(nav_chain))
    tracer.save(trace_path)

    already_learned = _check_knowledge_exists(app, signature)
    if already_learned and max_depth <= 0:
        print(f"  已学习过，跳过: {already_learned}")
        tracer.record_transition(_src, _tap, page_name, "skipped_known")
        tracer.save(trace_path)
        return None

    out_dir = app_log_dir / page_name
    node = DfsPageNode(
        page_name=page_name, signature=signature,
        parent=parent_page, via_tap=via_tap,
        depth=len(nav_chain), nav_chain=list(nav_chain),
        out_dir=out_dir,
    )

    # Probe (skip if max_depth reached)
    if max_depth == 0:
        print(f"  [max_depth=0] 跳过探测，仅记录页面")
        tracer.record_transition(_src, _tap, page_name, "depth_limit")
        tracer.save(trace_path)
        node.recon_result = None
    else:
        tracer.record_transition(_src, _tap, page_name, "entered")
        tracer.save(trace_path)
        try:
            # DFS-style incremental probing with callback
            def _on_element_tapped(area, after_bytes, navigated, tap_index, tap_result):
                """Called after each element tap. Enter child page if navigated."""
                print(f"    [DEBUG] _on_element_tapped called: area={area.label}, navigated={navigated}, max_depth={max_depth}")
                if not navigated or max_depth <= 0:
                    print(f"    [DEBUG] Skipping child entry (navigated={navigated}, max_depth={max_depth})")
                    return True  # Continue probing

                # Enter child page
                print(f"\n  → 进入子页面「{area.label}」")
                print(f"    [DEBUG] Entering child page: max_depth={max_depth}, will pass {max_depth - 1} to child")
                ax, ay = area.center_xy
                lx, ly = logical_xy(ax, ay)
                child_chain = nav_chain + [(lx, ly, area.label)]

                # Build child nav_stack with parent forward_coords
                parent_bytes, _ = nav_stack[-1]
                child_nav_stack = nav_stack[:-1] + [(parent_bytes, (lx, ly)), (after_bytes, None)]

                # Recursive DFS into child
                child_node = _dfs_recursive(
                    phone, child_chain, child_nav_stack, root_ctx,
                    chain_to_page, visited_screenshots, visited_page_entries, id_comp, tracer, trace_path,
                    app_log_dir, max_depth - 1, sample,
                )

                if child_node is not None:
                    node.children.append(child_node)

                # Return to current page
                print(f"\n  ← 返回「{page_name}」")
                ok, back_log = return_to_initial(
                    phone.client, phone.screenshot, nav_stack,
                    before_back_bytes=phone.screenshot(),
                )
                if not ok:
                    recovered = _manual_recover(
                        phone.client, phone.screenshot, nav_stack,
                        len(nav_stack) - 1,
                        prompt=f"子页面探索后无法返回 {page_name}",
                    )
                    if not recovered:
                        raise ProbeAbortedError(
                            f"DFS: 无法从子页面返回 {page_name}",
                            failed_tap=-1, failed_element="",
                            back_attempts=back_log,
                        )

                # Update tap result with back attempts
                tap_result.back_attempts = back_log

                return True  # Continue probing next element

            node.recon_result, node.out_dir = _probe_page_dfs(
                phone, knowledge, png_bytes, out_dir,
                sample=sample, nav_stack=nav_stack,
                on_element_tapped=_on_element_tapped,
            )
        except ProbeAbortedError as e:
            tracer.record_error(page_name, e)
            tracer.save(trace_path)
            node.error = {"message": str(e)}
            return node

    # POST: phone on this page
    return node


def generate_knowledge_postorder(nodes: list[DfsPageNode], app: str) -> None:
    """Walk tree post-order (children before parents), generate knowledge bottom-up."""
    from policy_expr.recon_cli import _check_knowledge_exists, run_knowledge, run_learn

    total = _count_nodes(nodes)
    generated = [0]

    def _visit(node: DfsPageNode):
        # Children first
        for child in node.children:
            _visit(child)

        # Then this node
        if node.recon_result and not node.knowledge_generated:
            already = _check_knowledge_exists(app, node.signature)
            if not already:
                run_learn(node.out_dir / "recon_result.json")
                run_knowledge(node.out_dir)
            node.knowledge_generated = True
            generated[0] += 1
            print(f"  [{generated[0]}/{total}] 知识生成: {node.page_name}")

    for node in nodes:
        _visit(node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_identity(phone) -> tuple:
    """Screenshot + parse page identity. Returns (png_bytes, knowledge, page_name)."""
    from policy_expr.recon.page_parser import PageParser
    from policy_expr.recon_cli import _derive_page_name

    png_bytes = phone.screenshot()
    knowledge = PageParser().analyze_screen(png_bytes)
    page_name = _derive_page_name(knowledge.page.identity.signature)
    return png_bytes, knowledge, page_name


def _probe_page_dfs(phone, knowledge, png_bytes, out_dir: Path,
                    sample: int = 0,
                    nav_stack: list | None = None,
                    on_element_tapped: callable | None = None) -> tuple:
    """DFS-style incremental probing: tap each element one by one, with callback after each tap.

    Args:
        on_element_tapped: callback(area, after_bytes, navigated, tap_index) -> bool
            Returns True to continue probing, False to stop.

    Returns (ReconResult, out_dir).
    """
    import random
    from policy_expr.executor import logical_xy
    from policy_expr.recon import viz_result
    from policy_expr.recon.utils import TapResult, ReconResult
    from llm.structured import get_llm_call_count

    llm_count_start = get_llm_call_count()

    page = knowledge.page
    areas = knowledge.areas

    # Skip back buttons (their behavior is known)
    areas = [a for a in areas if a.element_type != "back_button"]

    if sample > 0 and sample < len(areas):
        areas = random.sample(areas, sample)
        print(f"  [采样模式] 随机选取 {sample} 个元素")

    print(f"\n{'=' * 60}")
    print(f"点击探测: {len(areas)} 个可交互区域")
    print(f"{'=' * 60}")

    ident = page.identity
    result = ReconResult(
        app_name=ident.app_name,
        page_title=ident.page_title,
        page_type=ident.page_type,
        signature=ident.signature,
        description=page.description,
        elements_count=len(page.interactive_elements),
        initial_screenshot_path=str(out_dir / "initial.png"),
    )

    # Save initial screenshot
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "initial.png"
    img_path.write_bytes(png_bytes)
    viz_result(knowledge, png_bytes, "initial", out_dir)

    tap_dir = out_dir / "tap"
    tap_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "recon_result.json"

    has_nav = page.bottom_nav.has_nav
    top_level = len(nav_stack) - 1 if nav_stack else 0

    for i, area in enumerate(areas, 1):
        ax, ay = area.center_xy
        lx, ly = logical_xy(ax, ay)
        is_tab = has_nav and ay > 900
        print(f"\n  [{i}/{len(areas)}] 「{area.label}」 @ ({ax:.0f},{ay:.0f}) → ({lx:.0f},{ly:.0f})")

        tap_response = phone.client.tap(lx, ly)
        if "paused" in tap_response.lower():
            print(f"    Mac 弹窗阻断，关闭后跳过")
            try_resume_mac(phone.client)
            result.taps.append(TapResult(
                index=i, element_type="tab" if is_tab else "area",
                label=area.label, x=ax, y=ay,
                tap_ok=True, screenshot_path="", navigated=False,
            ))
            result.save(result_path)
            continue

        tap_ok = "failed" not in tap_response.lower() and "interrupted" not in tap_response.lower()
        print(f"    结果: {tap_response}")
        time.sleep(2.0)

        after_bytes = phone.screenshot()
        after_path = tap_dir / f"tap_{i:02d}_{area.label}.png"
        if after_bytes:
            after_path.write_bytes(after_bytes)
            print(f"    截图: {after_path}")

        navigated = False
        if after_bytes and nav_stack:
            from policy_expr.recon.back_nav import _get_change_comp
            comp = _get_change_comp()
            nav_result, nav_reason = comp.detect_navigation(nav_stack[top_level][0], after_bytes)
            if not nav_result:
                print(f"    {nav_reason}，跳过")
            else:
                navigated = True

        tap_result = TapResult(
            index=i,
            element_type="tab" if is_tab else "area",
            label=area.label,
            x=ax,
            y=ay,
            tap_ok=tap_ok,
            screenshot_path=str(after_path),
            navigated=navigated,
        )
        result.taps.append(tap_result)

        # Callback for DFS to enter child pages
        if on_element_tapped:
            should_continue = on_element_tapped(area, after_bytes, navigated, i, tap_result)
            if not should_continue:
                print(f"    [DFS] 探测中断（callback 返回 False）")
                break

    # Final save after all taps and callbacks complete
    result.save(result_path)
    ok = sum(1 for t in result.taps if t.tap_ok and t.navigated)
    llm_used = get_llm_call_count() - llm_count_start
    print(f"  探测完成: {len(result.taps)} 个元素 (成功导航 {ok}, LLM 调用 {llm_used} 次)")
    return result, out_dir


def _check_knowledge_exists(app: str, signature: str) -> Path | None:
    from policy_expr.recon_cli import _check_knowledge_exists as _cek
    return _cek(app, signature)


def _count_nodes(nodes: list[DfsPageNode]) -> int:
    return sum(1 + _count_nodes(n.children) for n in nodes)


# ---------------------------------------------------------------------------
# Post-exploration similarity report
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(
    page_entries: list[tuple[str, bytes]],
    threshold: float = 0.5,
    save_path: Path | None = None,
) -> list[dict]:
    """Compute pairwise similarity for all visited pages and print a report.

    Args:
        page_entries: list of (page_name, png_bytes) in visit order.
        threshold: pairs with similarity >= threshold are flagged in the report.
        save_path: if given, save the full pair list as JSON.

    Returns:
        list of {"page_a", "page_b", "similarity"} sorted by similarity desc.
    """
    import json
    from policy_expr.recon.page_compare import PageComparator

    comparator = PageComparator()
    n = len(page_entries)
    pairs: list[dict] = []

    for i in range(n):
        for j in range(i + 1, n):
            name_a, bytes_a = page_entries[i]
            name_b, bytes_b = page_entries[j]
            sim = comparator.raw_similarity(bytes_a, bytes_b)
            pairs.append({"page_a": name_a, "page_b": name_b, "similarity": round(sim, 4)})

    pairs.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"页面两两相似度报告  共 {n} 个页面 / {len(pairs)} 对  阈值={threshold}")
    print(f"{'=' * 70}")
    for p in pairs:
        flag = "  ⚠ 高相似" if p["similarity"] >= threshold else ""
        print(f"  {p['page_a'][:24]:24s}  vs  {p['page_b'][:24]:24s}  {p['similarity']:.4f}{flag}")

    above = [p for p in pairs if p["similarity"] >= threshold]
    print(f"\n  >= {threshold}: {len(above)} 对 / {len(pairs)} 对")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps({"threshold": threshold, "pairs": pairs}, ensure_ascii=False, indent=2))
        print(f"  已保存: {save_path}")

    return pairs
