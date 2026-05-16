"""
Experiment: LLM text fingerprint + semantic similarity for page deduplication.

Flow:
  screenshot → Claude → "page fingerprint text" → sentence-transformer embedding
  → cosine similarity matrix → check if same-type pages cluster together

Test subset: pages with known duplicates from 微信 dataset.
"""

import json
import base64
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer

DATASET_DIR = Path("logs/dataset/微信")
PAGES_DIR = DATASET_DIR / "pages"
OUTPUT_FILE = DATASET_DIR / "semantic_exp_results.json"

# Pages with duplicates — a representative subset covering different page types
SELECTED_GROUPS = {
    "搜索页":   [1, 3, 17, 21, 40, 65],
    "聊天列表": [0, 33, 54, 78],
    "发现页":   [16, 35, 50, 100],
    "聊天窗口": [34, 47, 87, 90, 92, 108],
    "服务通知": [20, 22, 23, 30, 39],
    "订单详情": [24, 25, 26, 29, 71],
    "聊天详情": [41, 43, 44, 74, 85, 89],
    "个人主页": [14, 38, 60, 93],
}

FINGERPRINT_PROMPT = """为这个 App 截图生成"页面模板描述"，目标是：同一类型页面（内容不同、结构相同）的描述尽量接近；不同类型页面的描述尽量有区别。

严格按以下格式输出三行，不要添加任何其他内容：
用途：<这类页面让用户做什么，动宾短语，对该类型所有实例都成立>
内容区：<主内容区的数据组织形式，描述结构而非具体数据，如"可滚动的会话条目列表"/"按时间排列的气泡对话流"/"单一对象的字段详情卡">
交互方式：<用户与页面的主要交互，如"滚动浏览并点击进入详情"/"输入文字并发送消息"/"切换开关配置选项"/"全屏查看并左右翻页">

规则（违反则描述无效）：
- 假设页面换了不同用户的数据，你的描述仍然成立——不能描述任何具体内容（特定商品、消息正文、金额、用户名）
- 不提状态栏、顶部标题/返回按钮、底部固定导航栏——这些每页都有，不具备区分价值
- 全部中文，每行不超过 25 字"""


def load_dataset_index() -> dict[int, dict]:
    with open(DATASET_DIR / "dataset.json") as f:
        data = json.load(f)
    return {p["idx"]: p for p in data["pages"]}


def encode_screenshot(page_dir_name: str) -> str:
    img_path = PAGES_DIR / page_dir_name / "screenshot.png"
    with open(img_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen3.5-flash"


def generate_fingerprint(client: OpenAI, page_dir_name: str) -> str:
    img_b64 = encode_screenshot(page_dir_name)
    response = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        max_tokens=200,
        extra_body={"enable_thinking": False},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": FINGERPRINT_PROMPT},
            ],
        }],
    )
    return response.choices[0].message.content.strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=DASHSCOPE_BASE_URL,
    )
    index = load_dataset_index()

    # Collect all selected page indices with their group label
    selected: list[tuple[int, str]] = []
    for group_name, idxs in SELECTED_GROUPS.items():
        for idx in idxs:
            selected.append((idx, group_name))

    selected.sort(key=lambda x: x[0])
    print(f"Selected {len(selected)} pages across {len(SELECTED_GROUPS)} groups\n")

    # Load cache if exists
    cache_file = DATASET_DIR / "fingerprint_cache.json"
    cache: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached fingerprints")

    # Generate fingerprints
    records = []
    for i, (idx, group) in enumerate(selected):
        page = index[idx]
        dir_name = page["dir_name"]
        key = dir_name

        if key in cache:
            fingerprint = cache[key]
            print(f"  [{i+1:02d}/{len(selected)}] [cache] {idx:03d} {page['name']}: {fingerprint[:55]}...")
        else:
            print(f"  [{i+1:02d}/{len(selected)}] [llm]   {idx:03d} {page['name']} ... ", end="", flush=True)
            fingerprint = generate_fingerprint(client, dir_name)
            cache[key] = fingerprint
            # Write cache immediately after each LLM call
            with open(cache_file, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"{fingerprint[:55]}...")

        records.append({
            "idx": idx,
            "name": page["name"],
            "group": group,
            "dir_name": dir_name,
            "fingerprint": fingerprint,
        })

    print(f"\nAll {len(cache)} fingerprints cached → {cache_file}")

    # Embed all fingerprints
    print("\nLoading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-zh")
    texts = [r["fingerprint"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute cosine similarity matrix
    n = len(records)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_sim(embeddings[i], embeddings[j])

    # === 1. 基础分布 ===
    print("\n=== 相似度分布 ===")
    same_sims, diff_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i][j]
            if records[i]["group"] == records[j]["group"]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)
    print(f"同组对:  {len(same_sims):3d}  avg={np.mean(same_sims):.3f}  min={np.min(same_sims):.3f}  max={np.max(same_sims):.3f}")
    print(f"跨组对:  {len(diff_sims):3d}  avg={np.mean(diff_sims):.3f}  min={np.min(diff_sims):.3f}  max={np.max(diff_sims):.3f}")
    print(f"间隔:    {np.mean(same_sims) - np.mean(diff_sims):.3f}")

    # === 2. 最近邻准确率 ===
    print("\n=== 最近邻准确率（NN Accuracy）===")
    # 对每个页面，找最相似的邻居，看是否同组
    nn_correct = 0
    nn_details = []
    for i in range(n):
        sims_to_others = [(sim_matrix[i][j], j) for j in range(n) if j != i]
        nn_sim, nn_j = max(sims_to_others)
        same = records[i]["group"] == records[nn_j]["group"]
        if same:
            nn_correct += 1
        nn_details.append({"i": i, "nn_j": nn_j, "nn_sim": nn_sim, "correct": same})

    print(f"NN 准确率: {nn_correct}/{n} = {nn_correct/n:.1%}")
    print("\n错误案例（最近邻是不同组）:")
    for d in nn_details:
        if not d["correct"]:
            ri, rj = records[d["i"]], records[d["nn_j"]]
            print(f"  [{ri['idx']:03d}]{ri['group']} → NN=[{rj['idx']:03d}]{rj['group']}  sim={d['nn_sim']:.3f}")

    # === 3. 探索仿真 ===
    # 模拟真实探索流程：按顺序到达页面，和库里已有页面比较
    # 库里最大相似度 > threshold → 跳过（认为已见过）
    # 评估：
    #   正确跳过 = 跳过 且 库里有同组页面 (TP)
    #   错误跳过 = 跳过 且 库里无同组页面 (FP，丢失新页面！)
    #   正确新增 = 未跳过 且 库里无同组页面 (TN)
    #   漏报重复 = 未跳过 且 库里已有同组页面 (FN)
    print("\n=== 探索仿真（随机顺序 × 20 次取均值）===")

    import random
    N_TRIALS = 20

    for threshold in [0.75, 0.80, 0.85, 0.90]:
        tp_list, fp_list, fn_list = [], [], []
        for _ in range(N_TRIALS):
            order = list(range(n))
            random.shuffle(order)
            library: list[int] = []  # indices of pages added to library

            tp = fp = fn = 0
            for i in order:
                if not library:
                    library.append(i)
                    continue
                max_sim = max(sim_matrix[i][j] for j in library)
                lib_groups = {records[j]["group"] for j in library}
                has_same_in_lib = records[i]["group"] in lib_groups

                if max_sim > threshold:  # 判断为已见，跳过
                    if has_same_in_lib:
                        tp += 1  # 正确跳过
                    else:
                        fp += 1  # 错误跳过（漏掉新页面）
                else:             # 判断为新页面，加入库
                    library.append(i)
                    if has_same_in_lib:
                        fn += 1  # 漏报重复（重复进库）

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        avg_tp = np.mean(tp_list)
        avg_fp = np.mean(fp_list)
        avg_fn = np.mean(fn_list)
        prec = avg_tp / (avg_tp + avg_fp) if (avg_tp + avg_fp) > 0 else 0
        rec  = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        print(f"  threshold={threshold:.2f}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}"
              f"  正确跳过={avg_tp:.1f}  错误跳过={avg_fp:.1f}  漏报={avg_fn:.1f}")

    # === GUIClip 视觉 embedding ===
    print("\n=== GUIClip 视觉相似度 ===")
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    guiclip_name = "Jl-wei/guiclip-vit-base-patch32"
    print(f"Loading {guiclip_name} ...")
    processor = CLIPProcessor.from_pretrained(guiclip_name, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(guiclip_name, local_files_only=True)
    clip_model.eval()

    vis_embs = []
    for r in records:
        img_path = PAGES_DIR / r["dir_name"] / "screenshot.png"
        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            pixel = {k: v for k, v in inputs.items() if k.startswith("pixel")}
            emb = clip_model.visual_projection(clip_model.vision_model(**pixel).pooler_output)
            emb = F.normalize(emb, p=2, dim=1).squeeze(0).numpy()
        vis_embs.append(emb)
    vis_embs = np.array(vis_embs)

    vis_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vis_matrix[i][j] = float(np.dot(vis_embs[i], vis_embs[j]))

    vis_same = [vis_matrix[i][j] for i in range(n) for j in range(i+1, n) if records[i]["group"] == records[j]["group"]]
    vis_diff = [vis_matrix[i][j] for i in range(n) for j in range(i+1, n) if records[i]["group"] != records[j]["group"]]
    print(f"视觉同组:  avg={np.mean(vis_same):.3f}  min={np.min(vis_same):.3f}  max={np.max(vis_same):.3f}")
    print(f"视觉跨组:  avg={np.mean(vis_diff):.3f}  min={np.min(vis_diff):.3f}  max={np.max(vis_diff):.3f}")
    print(f"视觉间隔:  {np.mean(vis_same) - np.mean(vis_diff):.3f}")

    # === 融合：text + visual ===
    print("\n=== 融合评估（text α + visual (1-α)）===")

    def run_sim_eval(mat: np.ndarray, label: str) -> None:
        s_same = [mat[i][j] for i in range(n) for j in range(i+1, n) if records[i]["group"] == records[j]["group"]]
        s_diff = [mat[i][j] for i in range(n) for j in range(i+1, n) if records[i]["group"] != records[j]["group"]]
        gap = np.mean(s_same) - np.mean(s_diff)

        best_f1, best_t = 0.0, 0.0
        for t in [0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
            tp_list, fp_list, fn_list = [], [], []
            for _ in range(20):
                order = list(range(n))
                random.shuffle(order)
                library: list[int] = []
                tp = fp = fn = 0
                for i in order:
                    if not library:
                        library.append(i)
                        continue
                    max_s = max(mat[i][j] for j in library)
                    has_same = records[i]["group"] in {records[j]["group"] for j in library}
                    if max_s > t:
                        (tp if has_same else fp).__add__(1)  # workaround for readability
                        if has_same: tp += 1
                        else: fp += 1
                    else:
                        library.append(i)
                        if has_same: fn += 1
                tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)
            atp, afp, afn = np.mean(tp_list), np.mean(fp_list), np.mean(fn_list)
            p = atp/(atp+afp) if (atp+afp) > 0 else 0
            r = atp/(atp+afn) if (atp+afn) > 0 else 0
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t
                best_p, best_r = p, r
        print(f"  {label:30s}  间隔={gap:.3f}  最佳F1={best_f1:.3f} (t={best_t:.2f}, P={best_p:.2f}, R={best_r:.2f})")

    run_sim_eval(sim_matrix, "text only")
    run_sim_eval(vis_matrix, "visual only (GUIClip)")
    for alpha in [0.3, 0.5, 0.7]:
        fused = alpha * sim_matrix + (1 - alpha) * vis_matrix
        run_sim_eval(fused, f"text*{alpha:.1f} + visual*{1-alpha:.1f}")

    # === 级联方案：先视觉，再语义 ===
    # 逻辑：visual_sim > tv → skip（不用 LLM）
    #       visual_sim <= tv 且 text_sim > tt → skip（LLM 兜底）
    #       否则 → 新页面
    print("\n=== 级联方案（visual OR text）===")
    print(f"  {'tv':>5}  {'tt':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'LLM省略率':>10}")

    best_f1, best_cfg = 0.0, None
    for tv in [0.80, 0.85, 0.90, 0.95]:
        for tt in [0.85, 0.88, 0.90, 0.92, 0.95]:
            tp_list, fp_list, fn_list, saved_list = [], [], [], []
            for _ in range(N_TRIALS):
                order = list(range(n))
                random.shuffle(order)
                library: list[int] = []
                tp = fp = fn = llm_saved = 0
                for i in order:
                    if not library:
                        library.append(i)
                        continue
                    vis_max = max(vis_matrix[i][j] for j in library)
                    has_same = records[i]["group"] in {records[j]["group"] for j in library}

                    if vis_max > tv:
                        llm_saved += 1  # 视觉命中，不需要 LLM
                        if has_same: tp += 1
                        else: fp += 1
                    else:
                        txt_max = max(sim_matrix[i][j] for j in library)
                        if txt_max > tt:
                            if has_same: tp += 1
                            else: fp += 1
                        else:
                            library.append(i)
                            if has_same: fn += 1

                total_decisions = tp + fp + fn + len(library) - 1
                tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)
                saved_list.append(llm_saved / max(total_decisions, 1))

            atp, afp, afn = np.mean(tp_list), np.mean(fp_list), np.mean(fn_list)
            p = atp/(atp+afp) if (atp+afp) > 0 else 0
            r = atp/(atp+afn) if (atp+afn) > 0 else 0
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
            saved_rate = np.mean(saved_list)
            if f1 > best_f1:
                best_f1, best_cfg = f1, (tv, tt, p, r, saved_rate)
            if f1 >= 0.85:
                print(f"  tv={tv:.2f}  tt={tt:.2f}  P={p:.2f}  R={r:.2f}  F1={f1:.3f}  省LLM={saved_rate:.1%}")

    if best_cfg:
        tv, tt, p, r, sr = best_cfg
        print(f"\n最佳: tv={tv:.2f} tt={tt:.2f}  F1={best_f1:.3f}  P={p:.2f}  R={r:.2f}  省LLM={sr:.1%}")

    # Save results
    results = {
        "records": records,
        "sim_matrix": sim_matrix.tolist(),
        "stats": {
            "same_avg": float(np.mean(same_sims)),
            "diff_avg": float(np.mean(diff_sims)),
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
