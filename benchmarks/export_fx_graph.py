"""
FX Graph 导出与分析
====================
将 MiniTransformerBlock 导出为 PyTorch FX 计算图，
识别可融合的子图模式（attention pattern），
并映射到 MLIR 概念，为编译优化 Pass 提供依据。

输出：
  1. 控制台打印完整 FX Graph 结构（节点列表）。
  2. 注意力子图可融合模式分析。
  3. FX → MLIR 算子映射表。
  4. Graphviz DOT 可视化文件（若 graphviz 可用）。
  5. Markdown 分析报告 → reports/fx_graph_analysis.md

用法：
    python benchmarks/export_fx_graph.py [--hidden_size 768] [--num_heads 12] \
           [--seq_len 128] [--batch_size 1] [--export_dot]
"""

import argparse
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.fx as fx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mini_transformer import MiniTransformerBlock, TransformerConfig


# =====================================================================
# FX 算子 → MLIR 方言映射表
# =====================================================================
FX_TO_MLIR_MAP = OrderedDict([
    # --- linalg 方言（线性代数运算） ---
    ("aten.mm",               "linalg.matmul"),
    ("aten.bmm",              "linalg.batch_matmul"),
    ("aten.matmul",           "linalg.matmul / linalg.batch_matmul"),
    ("aten.linear",           "linalg.matmul + arith.addf（带 bias）"),
    ("aten.addmm",            "linalg.matmul + arith.addf"),

    # --- arith 方言（逐元素算术） ---
    ("aten.add",              "arith.addf"),
    ("aten.mul",              "arith.mulf"),
    ("aten.div",              "arith.divf"),
    ("aten.sub",              "arith.subf"),
    ("aten.neg",              "arith.negf"),
    ("aten.rsqrt",            "math.rsqrt"),

    # --- math 方言（数学函数） ---
    ("aten.exp",              "math.exp"),
    ("aten.log",              "math.log"),
    ("aten.tanh",             "math.tanh"),
    ("aten.sqrt",             "math.sqrt"),

    # --- 激活函数（可映射到 linalg.generic + math） ---
    ("aten.gelu",             "linalg.generic { math.erf + arith.mulf }"),
    ("aten.relu",             "arith.maximumf(x, 0)"),
    ("aten.silu",             "arith.mulf(x, sigmoid(x))"),
    ("aten.softmax",          "linalg.generic { math.exp, arith.divf }（归约）"),
    ("aten._softmax",         "linalg.generic { math.exp, arith.divf }（归约）"),

    # --- 归一化 ---
    ("aten.layer_norm",       "linalg.generic { mean + variance + normalize }"),
    ("aten.native_layer_norm","linalg.generic { mean + variance + normalize }"),

    # --- tensor 方言（形状变换） ---
    ("aten.view",             "tensor.collapse_shape / tensor.expand_shape"),
    ("aten.reshape",          "tensor.collapse_shape / tensor.expand_shape"),
    ("aten.transpose",        "linalg.transpose"),
    ("aten.permute",          "linalg.transpose（多维）"),
    ("aten.contiguous",       "memref.copy（如需布局转换）"),
    ("aten.unbind",           "tensor.extract_slice ×N"),
    ("aten.cat",              "tensor.insert_slice"),
    ("aten.slice",            "tensor.extract_slice"),
    ("aten.select",           "tensor.extract_slice"),

    # --- 遮罩相关 ---
    ("aten.masked_fill",      "arith.select + arith.constant"),
    ("aten.triu",             "linalg.generic { arith.cmpi + arith.select }"),
    ("aten.where",            "arith.select"),

    # --- memref 方言（内存操作） ---
    ("aten.clone",            "memref.copy"),
    ("aten.copy_",            "memref.copy"),
    ("aten.to",               "arith.truncf / arith.extf（dtype 转换）"),

    # --- 特殊 ---
    ("aten.dropout",          "（推理时消除 → 恒等映射）"),
    ("aten.scaled_dot_product_attention",
     "custom.fused_attention（融合 kernel）"),
])


# =====================================================================
# 可融合子图模式定义
# =====================================================================
FUSION_PATTERNS = [
    {
        "name": "Attention 子操作融合（scale + mask + softmax）",
        "keywords": ["matmul", "mul", "div", "triu", "masked_fill", "softmax", "_softmax"],
        "description": "QK^T → scale → causal_mask → softmax 可融合为单个 fused kernel",
        "mlir_target": "custom.fused_scaled_masked_softmax",
    },
    {
        "name": "LayerNorm + 残差 Add 融合",
        "keywords": ["layer_norm", "native_layer_norm", "add"],
        "description": "LayerNorm 和后续的残差 add 可融合，减少一次全局读写",
        "mlir_target": "custom.fused_layer_norm_residual",
    },
    {
        "name": "MLP 激活融合（GeLU + bias）",
        "keywords": ["gelu", "add", "linear"],
        "description": "线性层 + GeLU 激活可融合为单个 kernel",
        "mlir_target": "custom.fused_linear_gelu",
    },
    {
        "name": "QKV 投影融合",
        "keywords": ["linear", "view", "unbind", "transpose"],
        "description": "QKV 投影 + reshape + split 可在一个 kernel 中完成",
        "mlir_target": "custom.fused_qkv_projection",
    },
]


# =====================================================================
# 工具函数
# =====================================================================

def get_op_name(node: fx.Node) -> str:
    """从 FX 节点提取人类可读的算子名称。"""
    if node.op == "call_function":
        target = str(node.target)
        # torch.ops.aten.xxx 形式
        if hasattr(node.target, "__name__"):
            return node.target.__name__
        if "aten" in target:
            return target.split(".")[-1].rstrip("'>)")
        return target.split(".")[-1].rstrip("'>)")
    elif node.op == "call_method":
        return f"Tensor.{node.target}"
    elif node.op == "call_module":
        return str(node.target)
    elif node.op == "get_attr":
        return f"attr:{node.target}"
    elif node.op == "placeholder":
        return f"input:{node.name}"
    elif node.op == "output":
        return "output"
    return str(node.op)


def classify_node(op_name: str) -> str:
    """将算子名称分类到功能组。"""
    op_lower = op_name.lower()

    if any(k in op_lower for k in ["matmul", "mm", "bmm", "linear", "addmm"]):
        return "MatMul（矩阵乘法）"
    if any(k in op_lower for k in ["softmax"]):
        return "Softmax"
    if any(k in op_lower for k in ["layer_norm", "native_layer_norm"]):
        return "LayerNorm"
    if any(k in op_lower for k in ["mask", "triu", "where", "select"]):
        return "Mask（遮罩）"
    if any(k in op_lower for k in ["gelu", "relu", "silu", "sigmoid"]):
        return "Activation（激活函数）"
    if any(k in op_lower for k in ["add", "mul", "div", "sub", "neg", "rsqrt"]):
        return "Elementwise（逐元素运算）"
    if any(k in op_lower for k in ["view", "reshape", "transpose", "permute",
                                     "contiguous", "unbind", "expand", "slice"]):
        return "Shape（形状变换）"
    if any(k in op_lower for k in ["clone", "copy", "to"]):
        return "Memory（内存操作）"
    if any(k in op_lower for k in ["dropout"]):
        return "Dropout"
    return "Other（其他）"


def detect_fusion_opportunities(nodes_info: list) -> list:
    """扫描节点列表，检测可融合子图模式。"""
    all_op_names = [n["op_name"].lower() for n in nodes_info]
    all_ops_str = " ".join(all_op_names)

    results = []
    for pattern in FUSION_PATTERNS:
        matched_keywords = [kw for kw in pattern["keywords"] if kw in all_ops_str]
        if len(matched_keywords) >= 2:  # 至少匹配 2 个关键词才算命中
            # 找到匹配的节点
            matched_nodes = []
            for n in nodes_info:
                if any(kw in n["op_name"].lower() for kw in pattern["keywords"]):
                    matched_nodes.append(n)

            results.append({
                "pattern": pattern,
                "matched_keywords": matched_keywords,
                "matched_nodes": matched_nodes,
                "node_count": len(matched_nodes),
            })
    return results


# =====================================================================
# 主流程
# =====================================================================

def export_and_analyze(cfg: TransformerConfig, export_dot: bool = False):
    """导出 FX Graph 并进行分析。"""

    print("=" * 70)
    print("  📦 FX Graph 导出与分析")
    print("=" * 70)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}")
    print(f"  seq_len     : {cfg.seq_len}")
    print(f"  batch_size  : {cfg.batch_size}")
    print("=" * 70)

    # ---- 方法 1：torch.fx.symbolic_trace ----
    print("\n📊 方法 1：torch.fx.symbolic_trace")
    print("-" * 50)

    model = MiniTransformerBlock(cfg, use_sdpa=False).to(cfg.device).to(cfg.dtype)
    model.eval()

    try:
        traced = fx.symbolic_trace(model)
        print("✅ symbolic_trace 成功。")
        print(f"\n{traced.graph}")
    except Exception as e:
        print(f"⚠️ symbolic_trace 失败（data-dependent 控制流）：{e}")
        traced = None

    # ---- 方法 2：torch.export（更强大，支持动态形状） ----
    print("\n📊 方法 2：torch.export（推荐）")
    print("-" * 50)

    model2 = MiniTransformerBlock(cfg, use_sdpa=False).to(cfg.device).to(cfg.dtype)
    model2.eval()
    example_input = torch.randn(
        cfg.batch_size, cfg.seq_len, cfg.hidden_size,
        device=cfg.device, dtype=cfg.dtype,
    )

    exported = None
    try:
        exported = torch.export.export(model2, (example_input,))
        print("✅ torch.export 成功。")
        print(f"\n{exported.graph}")
    except Exception as e:
        print(f"⚠️ torch.export 失败：{e}")
        # 回退到 torch._dynamo.export
        print("\n尝试回退到 torch._dynamo.export ...")
        try:
            from torch._dynamo import export as dynamo_export
            exported_gm, guards = dynamo_export(model2, example_input)
            print("✅ dynamo.export 成功。")
            print(f"\n{exported_gm.graph}")
            exported = exported_gm
        except Exception as e2:
            print(f"❌ dynamo.export 也失败：{e2}")

    # ---- 使用可用的 graph 进行分析 ----
    graph_to_analyze = None
    if exported is not None:
        if hasattr(exported, 'graph_module'):
            graph_to_analyze = exported.graph_module.graph
        elif hasattr(exported, 'graph'):
            graph_to_analyze = exported.graph
    elif traced is not None:
        graph_to_analyze = traced.graph

    if graph_to_analyze is None:
        print("\n❌ 无法获取计算图，分析终止。")
        return

    # ---- 节点分析 ----
    print("\n" + "=" * 70)
    print("  🔍 FX Graph 节点分析")
    print("=" * 70)

    nodes_info = []
    for node in graph_to_analyze.nodes:
        op_name = get_op_name(node)
        category = classify_node(op_name)
        info = {
            "name": node.name,
            "op": node.op,
            "op_name": op_name,
            "category": category,
            "num_users": len(node.users),
            "num_inputs": len(node.args) + len(node.kwargs),
        }
        nodes_info.append(info)

    # 打印节点表格
    print(f"\n{'序号':>4s}  {'节点名称':<35s}  {'算子':<30s}  {'类别':<20s}  {'用户数':>5s}")
    print("-" * 100)
    for i, n in enumerate(nodes_info):
        if n["op"] in ("placeholder", "output"):
            marker = "⬛"
        elif "MatMul" in n["category"]:
            marker = "🔴"
        elif n["category"] in ("Softmax", "Mask（遮罩）"):
            marker = "🟡"
        elif "LayerNorm" in n["category"]:
            marker = "🔵"
        elif "Activation" in n["category"]:
            marker = "🟢"
        elif "Shape" in n["category"]:
            marker = "⚪"
        else:
            marker = "⬜"
        print(f"{marker}{i:>3d}  {n['name']:<35s}  {n['op_name']:<30s}  {n['category']:<20s}  {n['num_users']:>5d}")

    # ---- 节点分类统计 ----
    print("\n" + "=" * 70)
    print("  📊 节点分类统计")
    print("=" * 70)

    category_counts = defaultdict(int)
    for n in nodes_info:
        if n["op"] not in ("placeholder", "output"):
            category_counts[n["category"]] += 1

    total_ops = sum(category_counts.values())
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total_ops * 100 if total_ops > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cat:<25s}  {count:>4d} ({pct:>5.1f}%)  {bar}")
    print(f"  {'总计':<25s}  {total_ops:>4d}")

    # ---- 融合机会检测 ----
    print("\n" + "=" * 70)
    print("  🔧 可融合子图模式检测")
    print("=" * 70)

    fusion_results = detect_fusion_opportunities(nodes_info)
    for i, fr in enumerate(fusion_results, 1):
        p = fr["pattern"]
        print(f"\n  模式 {i}：{p['name']}")
        print(f"    说明：{p['description']}")
        print(f"    匹配关键词：{', '.join(fr['matched_keywords'])}")
        print(f"    涉及节点数：{fr['node_count']}")
        print(f"    MLIR 目标：{p['mlir_target']}")
        print(f"    涉及节点：")
        for mn in fr["matched_nodes"][:8]:
            print(f"      - {mn['name']} ({mn['op_name']})")
        if len(fr["matched_nodes"]) > 8:
            print(f"      ... 及其他 {len(fr['matched_nodes']) - 8} 个节点")

    # ---- FX → MLIR 映射表 ----
    print("\n" + "=" * 70)
    print("  🗺️ FX 算子 → MLIR 方言映射（本模型涉及的算子）")
    print("=" * 70)

    used_ops = set()
    for n in nodes_info:
        op_lower = n["op_name"].lower()
        for key in FX_TO_MLIR_MAP:
            aten_name = key.replace("aten.", "")
            if aten_name in op_lower:
                used_ops.add(key)

    print(f"\n{'FX / ATen 算子':<40s}  {'MLIR 映射':<50s}")
    print("-" * 92)
    for op in FX_TO_MLIR_MAP:
        if op in used_ops:
            marker = "✅"
        else:
            marker = "  "
        print(f"  {marker} {op:<37s}  {FX_TO_MLIR_MAP[op]:<50s}")

    # ---- 导出 DOT 可视化 ----
    if export_dot:
        dot_path = Path("reports") / "fx_graph.dot"
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        _export_dot(nodes_info, graph_to_analyze, dot_path)

    # ---- 生成 Markdown 报告 ----
    _generate_report(nodes_info, category_counts, fusion_results, cfg)

    print("\n✅ FX Graph 分析完成。")


def _export_dot(nodes_info, graph, dot_path):
    """导出 Graphviz DOT 文件用于可视化。"""
    lines = ['digraph fx_graph {', '  rankdir=TB;', '  node [shape=box, style=filled];']

    color_map = {
        "MatMul（矩阵乘法）": "#ff6b6b",
        "Softmax": "#ffd93d",
        "Mask（遮罩）": "#ffd93d",
        "LayerNorm": "#6bcbff",
        "Activation（激活函数）": "#6bff6b",
        "Elementwise（逐元素运算）": "#c4c4c4",
        "Shape（形状变换）": "#e8e8e8",
        "Memory（内存操作）": "#ffb6c1",
        "Dropout": "#dcdcdc",
        "Other（其他）": "#f0f0f0",
    }

    for node in graph.nodes:
        op_name = get_op_name(node)
        category = classify_node(op_name)
        color = color_map.get(category, "#f0f0f0")
        label = f"{node.name}\\n{op_name}"
        lines.append(f'  "{node.name}" [label="{label}", fillcolor="{color}"];')

    for node in graph.nodes:
        for arg in node.all_input_nodes:
            lines.append(f'  "{arg.name}" -> "{node.name}";')

    lines.append("}")

    with open(dot_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n📊 DOT 文件已保存 → {dot_path}")
    print("   可用 graphviz 渲染：dot -Tpng reports/fx_graph.dot -o reports/fx_graph.png")


def _generate_report(nodes_info, category_counts, fusion_results, cfg):
    """生成 Markdown 分析报告。"""
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"fx_graph_analysis_{now}.md"
    latest_path = report_dir / "fx_graph_analysis_latest.md"

    total_ops = sum(category_counts.values())

    md = []
    md.append("# FX Graph 分析报告\n\n")
    md.append(f"**配置**: hidden_size={cfg.hidden_size}, num_heads={cfg.num_heads}, ")
    md.append(f"seq_len={cfg.seq_len}, batch_size={cfg.batch_size}\n\n")

    # 节点分类统计
    md.append("## 节点分类统计\n\n")
    md.append("| 类别 | 节点数 | 占比 |\n")
    md.append("| --- | ---: | ---: |\n")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total_ops * 100 if total_ops > 0 else 0
        md.append(f"| {cat} | {count} | {pct:.1f}% |\n")
    md.append(f"| **总计** | **{total_ops}** | |\n\n")

    # 完整节点列表
    md.append("## FX Graph 节点列表\n\n")
    md.append("| # | 节点名称 | 算子 | 类别 |\n")
    md.append("| ---: | --- | --- | --- |\n")
    for i, n in enumerate(nodes_info):
        if n["op"] not in ("placeholder", "output"):
            md.append(f'| {i} | `{n["name"]}` | `{n["op_name"]}` | {n["category"]} |\n')
    md.append("\n")

    # 融合机会
    md.append("## 可融合子图模式\n\n")
    for i, fr in enumerate(fusion_results, 1):
        p = fr["pattern"]
        md.append(f"### 模式 {i}：{p['name']}\n\n")
        md.append(f"- **说明**：{p['description']}\n")
        md.append(f"- **涉及节点数**：{fr['node_count']}\n")
        md.append(f"- **MLIR 目标**：`{p['mlir_target']}`\n")
        md.append(f"- **匹配关键词**：{', '.join(fr['matched_keywords'])}\n\n")

    # FX → MLIR 映射
    md.append("## FX → MLIR 算子映射\n\n")
    md.append("| FX / ATen 算子 | MLIR 方言映射 |\n")
    md.append("| --- | --- |\n")
    for op, mlir in FX_TO_MLIR_MAP.items():
        md.append(f"| `{op}` | `{mlir}` |\n")
    md.append("\n")

    # 编译优化建议
    md.append("## 编译优化建议\n\n")
    md.append("基于 FX Graph 结构分析，提出以下 MLIR Pass 设计方向：\n\n")
    md.append("| Pass 名称 | 输入模式 | 输出 | 预期收益 |\n")
    md.append("| --- | --- | --- | --- |\n")

    shape_count = category_counts.get("Shape（形状变换）", 0)
    for fr in fusion_results:
        p = fr["pattern"]
        md.append(f"| `{p['mlir_target']}` | {p['description']} | 单个 fused kernel | "
                  f"减少 {fr['node_count']-1} 次 kernel launch |\n")

    if shape_count > 0:
        md.append(f"| `canonicalize<view>` | {shape_count} 个 view/reshape 节点 | "
                  f"消除冗余形状变换 | 减少 memory copy |\n")

    md.append("\n---\n\n")
    md.append(f"*报告由 `benchmarks/export_fx_graph.py` 自动生成于 "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    content = "".join(md)
    for path in (report_path, latest_path):
        with open(path, "w") as f:
            f.write(content)

    print(f"\n📄 FX Graph 分析报告已保存：")
    print(f"   {report_path}")
    print(f"   {latest_path}")


# =====================================================================
# CLI 入口
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FX Graph 导出与分析")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--export_dot", action="store_true",
                   help="导出 Graphviz DOT 可视化文件")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TransformerConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    export_and_analyze(cfg, export_dot=args.export_dot)
