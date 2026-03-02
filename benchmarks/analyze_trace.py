"""
Trace 分析工具
================
解析 PyTorch Profiler 导出的 Chrome trace JSON 文件，
生成跨 baseline / SDPA / compiled 的对比摘要。

分析结果会同时输出到控制台和 Markdown 报告文件。

用法：
    python benchmarks/analyze_trace.py [--trace_dir traces] [--report_dir reports]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_trace(path: Path) -> List[dict]:
    """从 JSON 文件中加载 Chrome trace 事件。"""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def analyze_events(events: List[dict]) -> Dict:
    """从 trace 事件中提取关键指标。"""
    gpu_events = [
        e for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "cuda_runtime")
        and e.get("ph") == "X"
        and "dur" in e
    ]

    kernel_events = [e for e in gpu_events if e.get("cat") == "kernel"]

    total_kernel_time_us = sum(e["dur"] for e in kernel_events)
    kernel_count = len(kernel_events)

    # Kernel 时长分布
    durations = [e["dur"] for e in kernel_events]
    small_kernels = [d for d in durations if d < 50]
    medium_kernels = [d for d in durations if 50 <= d < 500]
    large_kernels = [d for d in durations if d >= 500]

    # 内存拷贝事件
    memcpy_events = [e for e in gpu_events if e.get("cat") == "gpu_memcpy"]
    total_memcpy_time_us = sum(e.get("dur", 0) for e in memcpy_events)

    # Kernel 名称频率统计
    kernel_names = defaultdict(lambda: {"count": 0, "total_us": 0})
    for e in kernel_events:
        name = e.get("name", "unknown")
        kernel_names[name]["count"] += 1
        kernel_names[name]["total_us"] += e["dur"]

    # 按时间排序的 Top kernel
    top_kernels = sorted(
        kernel_names.items(), key=lambda x: x[1]["total_us"], reverse=True
    )[:10]

    # ---- 按功能分类 kernel（用于瓶颈分析）----
    # 分类规则：根据 kernel 名称中的关键词匹配到对应功能类别
    categories = {
        "attention_softmax": {
            "keywords": ["softmax"],
            "label": "Softmax",
            "count": 0, "total_us": 0,
        },
        "attention_mask": {
            "keywords": ["masked_fill", "mask", "triu", "tril"],
            "label": "Mask（遮罩）",
            "count": 0, "total_us": 0,
        },
        "matmul": {
            "keywords": ["gemm", "cutlass", "cublas", "Kernel2"],
            "label": "MatMul（矩阵乘法）",
            "count": 0, "total_us": 0,
        },
        "layernorm": {
            "keywords": ["layer_norm", "layernorm"],
            "label": "LayerNorm",
            "count": 0, "total_us": 0,
        },
        "elementwise": {
            "keywords": ["elementwise", "vectorized_elementwise", "add",
                         "gelu", "CUDAFunctor"],
            "label": "Elementwise（逐元素运算）",
            "count": 0, "total_us": 0,
        },
        "memory_op": {
            "keywords": ["copy", "clone", "contiguous", "memcpy", "memset",
                         "fill"],
            "label": "内存操作（copy/clone）",
            "count": 0, "total_us": 0,
        },
        "flash_attention": {
            "keywords": ["flash_fwd", "flash_bwd", "flash_attention"],
            "label": "FlashAttention（融合）",
            "count": 0, "total_us": 0,
        },
        "triton_fused": {
            "keywords": ["triton_", "fused_scale_mask_softmax", "online_softmax"],
            "label": "Triton 融合 kernel",
            "count": 0, "total_us": 0,
        },
    }

    uncategorized = {"count": 0, "total_us": 0}
    for kname, kstats in kernel_names.items():
        matched = False
        name_lower = kname.lower()
        for cat_key, cat in categories.items():
            if any(kw.lower() in name_lower for kw in cat["keywords"]):
                cat["count"] += kstats["count"]
                cat["total_us"] += kstats["total_us"]
                matched = True
                break  # 每个 kernel 只归入第一个匹配的类别
        if not matched:
            uncategorized["count"] += kstats["count"]
            uncategorized["total_us"] += kstats["total_us"]

    # 只保留有数据的类别
    kernel_categories = {
        k: {"label": v["label"], "count": v["count"], "total_us": v["total_us"]}
        for k, v in categories.items() if v["count"] > 0
    }
    if uncategorized["count"] > 0:
        kernel_categories["other"] = {
            "label": "其他",
            "count": uncategorized["count"],
            "total_us": uncategorized["total_us"],
        }

    return {
        "total_kernel_time_us": total_kernel_time_us,
        "kernel_count": kernel_count,
        "small_kernels": len(small_kernels),
        "medium_kernels": len(medium_kernels),
        "large_kernels": len(large_kernels),
        "memcpy_count": len(memcpy_events),
        "memcpy_time_us": total_memcpy_time_us,
        "top_kernels": top_kernels,
        "avg_kernel_us": total_kernel_time_us / max(kernel_count, 1),
        "kernel_categories": kernel_categories,
    }


def format_comparison(results: Dict[str, Dict]) -> str:
    """生成对比报告的文本内容（同时用于控制台输出和保存文件）。"""
    lines = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("  📊 Trace 对比分析报告")
    lines.append("=" * 80)

    # 表头
    names = list(results.keys())
    col_w = 18
    header = f"  {'指标':<30s}" + "".join(f"{n:>{col_w}s}" for n in names)
    lines.append(header)
    lines.append("  " + "-" * (30 + col_w * len(names)))

    # 指标行
    metrics = [
        ("Kernel 总时间 (ms)", lambda r: f"{r['total_kernel_time_us'] / 1e3:.2f}"),
        ("Kernel 启动总次数", lambda r: f"{r['kernel_count']}"),
        ("Kernel 平均时长 (μs)", lambda r: f"{r['avg_kernel_us']:.1f}"),
        ("小 kernel (<50μs)", lambda r: f"{r['small_kernels']}"),
        ("中 kernel (50-500μs)", lambda r: f"{r['medium_kernels']}"),
        ("大 kernel (≥500μs)", lambda r: f"{r['large_kernels']}"),
        ("内存拷贝事件数", lambda r: f"{r['memcpy_count']}"),
        ("内存拷贝时间 (ms)", lambda r: f"{r['memcpy_time_us'] / 1e3:.2f}"),
    ]

    for label, fn in metrics:
        row = f"  {label:<30s}"
        for name in names:
            row += f"{fn(results[name]):>{col_w}s}"
        lines.append(row)

    # 各 trace 的 Top kernel
    for name in names:
        lines.append(f"\n  🔧 Top kernel — {name}")
        lines.append(f"    {'Kernel':<50s} {'调用次数':>8s} {'总时间(μs)':>12s} {'均值(μs)':>10s}")
        lines.append("    " + "-" * 80)
        for kname, kstats in results[name]["top_kernels"]:
            avg = kstats["total_us"] / max(kstats["count"], 1)
            display = kname[:48]
            lines.append(
                f"    {display:<50s} {kstats['count']:>8d} "
                f"{kstats['total_us']:>12.1f} {avg:>10.1f}"
            )

    # 加速比摘要
    if len(names) >= 2:
        base = results[names[0]]
        lines.append(f"\n  ⚡ 相对于 {names[0]} 的加速比：")
        for name in names[1:]:
            other = results[name]
            if base["total_kernel_time_us"] > 0:
                speedup = base["total_kernel_time_us"] / max(other["total_kernel_time_us"], 1)
                kernel_reduction = (
                    (base["kernel_count"] - other["kernel_count"])
                    / max(base["kernel_count"], 1)
                    * 100
                )
                lines.append(
                    f"    {name}: 加速 {speedup:.2f}x，"
                    f"kernel 启动次数变化 {kernel_reduction:+.0f}%"
                )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_markdown_report(results: Dict[str, Dict], trace_dir: Path) -> str:
    """生成 Markdown 格式的详细对比分析报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    names = list(results.keys())
    md = []

    md.append("# Trace 对比分析报告\n")
    md.append(f"> **生成时间**: {now}\n")
    md.append(f"> **Trace 目录**: `{trace_dir}`\n")
    md.append(f"> **对比版本**: {', '.join(names)}\n")
    md.append("---\n")

    # ---- 总体对比表格 ----
    md.append("## 1. 总体对比\n")
    md.append("| 指标 | " + " | ".join(names) + " |")
    md.append("| --- | " + " | ".join(["---:"] * len(names)) + " |")

    metrics = [
        ("Kernel 总时间 (ms)", lambda r: f"{r['total_kernel_time_us'] / 1e3:.2f}"),
        ("Kernel 启动总次数", lambda r: f"{r['kernel_count']}"),
        ("Kernel 平均时长 (μs)", lambda r: f"{r['avg_kernel_us']:.1f}"),
        ("小 kernel (<50μs)", lambda r: f"{r['small_kernels']}"),
        ("中 kernel (50–500μs)", lambda r: f"{r['medium_kernels']}"),
        ("大 kernel (≥500μs)", lambda r: f"{r['large_kernels']}"),
        ("内存拷贝事件数", lambda r: f"{r['memcpy_count']}"),
        ("内存拷贝时间 (ms)", lambda r: f"{r['memcpy_time_us'] / 1e3:.2f}"),
    ]
    for label, fn in metrics:
        row = f"| {label} | " + " | ".join(fn(results[n]) for n in names) + " |"
        md.append(row)

    md.append("")

    # ---- 加速比 ----
    if len(names) >= 2:
        base = results[names[0]]
        md.append("## 2. 加速比（相对于基线）\n")
        md.append(f"基线: **{names[0]}**\n")
        md.append("| 版本 | 加速比 | Kernel 启动次数变化 |")
        md.append("| --- | ---: | ---: |")
        for name in names[1:]:
            other = results[name]
            if base["total_kernel_time_us"] > 0:
                speedup = base["total_kernel_time_us"] / max(other["total_kernel_time_us"], 1)
                kernel_reduction = (
                    (base["kernel_count"] - other["kernel_count"])
                    / max(base["kernel_count"], 1)
                    * 100
                )
                md.append(f"| {name} | {speedup:.2f}x | {kernel_reduction:+.0f}% |")
        md.append("")

    # ---- 各版本 Top kernel ----
    section_num = 3
    for name in names:
        md.append(f"## {section_num}. Top Kernel — {name}\n")
        md.append("| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |")
        md.append("| --- | ---: | ---: | ---: |")
        for kname, kstats in results[name]["top_kernels"]:
            avg = kstats["total_us"] / max(kstats["count"], 1)
            # 截断过长的 kernel 名称
            display = kname[:80]
            md.append(
                f"| `{display}` | {kstats['count']} | "
                f"{kstats['total_us']:.1f} | {avg:.1f} |"
            )
        md.append("")
        section_num += 1

    # ---- 瓶颈分析模板 ----
    md.append(f"## {section_num}. 瓶颈分析与优化假设\n")
    md.append("### 观察到的瓶颈\n")

    # 自动生成瓶颈观察
    if names:
        base = results[names[0]]
        total_kernels = base["kernel_count"]
        small_ratio = base["small_kernels"] / max(total_kernels, 1) * 100
        md.append(f"- 基线共 **{total_kernels}** 次 kernel 启动")
        md.append(f"- 小 kernel (<50μs) 占比 **{small_ratio:.0f}%**"
                  f"（{base['small_kernels']} / {total_kernels}）")
        if base["memcpy_count"] > 0:
            md.append(f"- 存在 **{base['memcpy_count']}** 次 Host↔Device 内存拷贝事件"
                      f"（耗时 {base['memcpy_time_us'] / 1e3:.2f} ms）")
        else:
            md.append("- 未检测到 Host↔Device 内存拷贝事件")

        if len(names) >= 2:
            sdpa_name = names[1]
            sdpa = results[sdpa_name]
            reduction = total_kernels - sdpa["kernel_count"]
            md.append(f"- **{sdpa_name}** 减少了 **{reduction}** 次 kernel 启动"
                      f"（{reduction / max(total_kernels, 1) * 100:.0f}% 减少）")

    md.append("")

    # ---- 按功能分类的 Kernel 分布（基于 baseline）----
    if names:
        base = results[names[0]]
        cats = base.get("kernel_categories", {})
        if cats:
            md.append(f"### Kernel 功能分类（基于 {names[0]}）\n")
            md.append("| 类别 | 启动次数 | 总时间 (μs) | 时间占比 |")
            md.append("| --- | ---: | ---: | ---: |")
            total_us = base["total_kernel_time_us"]
            for cat_key, cat in sorted(cats.items(),
                                        key=lambda x: x[1]["total_us"],
                                        reverse=True):
                pct = cat["total_us"] / max(total_us, 1) * 100
                md.append(f"| {cat['label']} | {cat['count']} | "
                          f"{cat['total_us']:.1f} | {pct:.1f}% |")
            md.append("")

    md.append("### 编译优化假设\n")
    md.append('> 以下"当前状态"均基于 **baseline** 的实际 profiling 数据。\n')
    md.append("| 优化方向 | 当前状态（baseline 实测） | 预期效果 |")
    md.append("| --- | --- | --- |")

    # 基于真实数据构造优化假设
    if names:
        base = results[names[0]]
        cats = base.get("kernel_categories", {})
        total_us = base["total_kernel_time_us"]

        # 假设 1：Attention 子操作融合
        attn_cats = ["attention_softmax", "attention_mask", "attention_scale"]
        attn_count = sum(cats.get(c, {}).get("count", 0) for c in attn_cats)
        attn_us = sum(cats.get(c, {}).get("total_us", 0) for c in attn_cats)
        attn_pct = attn_us / max(total_us, 1) * 100
        if attn_count > 0:
            md.append(
                f"| Attention 子操作融合（scale + mask + softmax） | "
                f"共 {attn_count} 次启动，耗时 {attn_us:.0f}μs（占比 {attn_pct:.1f}%） | "
                f"合并为 1 个 fused kernel，消除 {attn_count - 1} 次 launch overhead |"
            )
        else:
            md.append(
                "| Attention 子操作融合（scale + mask + softmax） | "
                "未检测到独立 attention 子操作 kernel | 可能已被融合 |"
            )

        # 假设 2：Elementwise 融合
        ln_info = cats.get("layernorm", {})
        ew_info = cats.get("elementwise", {})
        ew_count = ln_info.get("count", 0) + ew_info.get("count", 0)
        ew_us = ln_info.get("total_us", 0) + ew_info.get("total_us", 0)
        ew_pct = ew_us / max(total_us, 1) * 100
        if ew_count > 0:
            md.append(
                f"| Elementwise 融合（LayerNorm + Add + GeLU） | "
                f"共 {ew_count} 次启动，耗时 {ew_us:.0f}μs（占比 {ew_pct:.1f}%） | "
                f"合并为 2–3 个 fused kernel，减少显存读写 |"
            )
        else:
            md.append(
                "| Elementwise 融合（LayerNorm + Add + GeLU） | "
                "未检测到独立 elementwise kernel | 可能已被融合 |"
            )

        # 假设 3：内存拷贝消除
        memcpy_count = base["memcpy_count"]
        memcpy_us = base["memcpy_time_us"]
        if memcpy_count > 0:
            md.append(
                f"| 消除不必要的内存拷贝 | "
                f"共 {memcpy_count} 次 HtoD/DtoH，耗时 {memcpy_us:.0f}μs | "
                f"通过 buffer 复用消除冗余拷贝 |"
            )
        else:
            md.append(
                "| 消除不必要的内存拷贝 | "
                "未检测到 Host↔Device 拷贝 | 无需优化 |"
            )

        # 假设 4：内存操作融合
        mem_info = cats.get("memory_op", {})
        if mem_info.get("count", 0) > 0:
            mem_pct = mem_info["total_us"] / max(total_us, 1) * 100
            md.append(
                f"| 内存操作融合（copy/clone/contiguous） | "
                f"共 {mem_info['count']} 次启动，耗时 {mem_info['total_us']:.0f}μs"
                f"（占比 {mem_pct:.1f}%） | "
                f"通过 layout 变换消除冗余拷贝 |"
            )

    md.append("")

    md.append("---\n")
    md.append(f"*报告由 `benchmarks/analyze_trace.py` 自动生成于 {now}*\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="分析 profiling trace 文件")
    parser.add_argument("--trace_dir", type=str, default="traces")
    parser.add_argument("--report_dir", type=str, default="reports",
                        help="分析报告输出目录")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"❌ Trace 目录 '{trace_dir}' 不存在。")
        print("   请先运行 profiling 脚本。")
        sys.exit(1)

    trace_files = sorted(trace_dir.glob("*_trace.json"))
    if not trace_files:
        print(f"❌ 在 '{trace_dir}' 中未找到 *_trace.json 文件。")
        print("   请先运行 profiling 脚本：")
        print("     python benchmarks/profile_attention.py")
        print("     python benchmarks/profile_flash_attn.py")
        print("     python benchmarks/profile_compiled.py")
        print("     python benchmarks/profile_triton.py")
        sys.exit(1)

    print(f"📂 找到 {len(trace_files)} 个 trace 文件：")
    for f in trace_files:
        print(f"   {f}")

    # 分析每个 trace 文件
    results = {}
    for tf in trace_files:
        label = tf.stem.replace("_trace", "")
        events = load_trace(tf)
        results[label] = analyze_events(events)

    # 控制台输出
    console_text = format_comparison(results)
    print(console_text)

    # 保存 Markdown 报告
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"trace_analysis_{timestamp}.md"
    # 同时维护一个 latest 链接
    latest_path = report_dir / "trace_analysis_latest.md"

    md_content = generate_markdown_report(results, trace_dir)

    report_path.write_text(md_content, encoding="utf-8")
    latest_path.write_text(md_content, encoding="utf-8")

    print(f"\n📄 分析报告已保存：")
    print(f"   {report_path}")
    print(f"   {latest_path}  （最新版本）")


if __name__ == "__main__":
    main()
