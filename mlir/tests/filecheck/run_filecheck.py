"""
轻量级 MLIR FileCheck 测试框架
================================

实现 MLIR 风格的 CHECK 指令，用于验证 Pass 变换后的 IR 输出。

支持的指令：
  CHECK:        行中应包含指定文本
  CHECK-NOT:    行中不应包含指定文本
  CHECK-SAME:   在与上一个 CHECK 匹配的同一行中应包含指定文本
  CHECK-NEXT:   在上一个 CHECK 匹配行的下一行应包含指定文本
  CHECK-COUNT-N: 指定文本应在 IR 中出现恰好 N 次

用法:
  filecheck/basic_fusion.mlir:
    // RUN: attention_fusion
    // CHECK: custom.fused_scaled_masked_softmax
    // CHECK-SAME: is_causal = true
    // CHECK-NOT: torch.aten.softmax.int
    func.func @forward(%arg0: !torch.vtensor<[2,8,128,128],f32>)
        -> !torch.vtensor<[2,8,128,128],f32> { ... }

  python -m mlir.tests.filecheck.run_filecheck mlir/tests/filecheck/basic_fusion.mlir
"""

import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from torch_mlir import ir, rewrite
from torch_mlir.dialects import torch as torch_dialect

from mlir.passes.attention_fusion_pass import (
    run_attention_fusion_pass,
    create_attention_fusion_patterns,
)
from mlir.passes.incremental_softmax_pass import (
    run_online_softmax_pass,
    decompose_softmax,
    create_online_softmax_patterns,
)
from mlir.passes.pass_pipeline import (
    build_attention_optimization_pipeline,
    build_online_softmax_pipeline,
)


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class CheckDirective:
    """解析后的 CHECK 指令。"""
    kind: str          # "CHECK", "CHECK-NOT", "CHECK-SAME", "CHECK-NEXT", "CHECK-COUNT"
    pattern: str       # 要匹配的文本
    line_num: int      # 源文件行号（用于错误报告）
    count: int = 0     # CHECK-COUNT-N 的 N

@dataclass
class CheckResult:
    """单个 CHECK 指令的验证结果。"""
    directive: CheckDirective
    passed: bool
    message: str = ""


# =====================================================================
# 解析
# =====================================================================

def parse_directives(mlir_text: str) -> tuple:
    """
    从 .mlir 文件解析 CHECK 指令和 RUN 指令。

    Returns:
        (run_pass_name, ir_body, directives)
        - run_pass_name: RUN 指令指定的 pass 名
        - ir_body: 去除 CHECK 注释后的 MLIR IR（用于解析）
        - directives: CheckDirective 列表
    """
    directives: List[CheckDirective] = []
    run_pass = None
    ir_lines = []

    for line_num, line in enumerate(mlir_text.split("\n"), 1):
        stripped = line.strip()

        # RUN 指令
        m = re.match(r'//\s*RUN:\s*(.+)', stripped)
        if m:
            run_pass = m.group(1).strip()
            continue

        # CHECK-COUNT-N
        m = re.match(r'//\s*CHECK-COUNT-(\d+):\s*(.+)', stripped)
        if m:
            directives.append(CheckDirective(
                kind="CHECK-COUNT", pattern=m.group(2).strip(),
                line_num=line_num, count=int(m.group(1)),
            ))
            continue

        # CHECK-NOT
        m = re.match(r'//\s*CHECK-NOT:\s*(.+)', stripped)
        if m:
            directives.append(CheckDirective(
                kind="CHECK-NOT", pattern=m.group(1).strip(),
                line_num=line_num,
            ))
            continue

        # CHECK-SAME
        m = re.match(r'//\s*CHECK-SAME:\s*(.+)', stripped)
        if m:
            directives.append(CheckDirective(
                kind="CHECK-SAME", pattern=m.group(1).strip(),
                line_num=line_num,
            ))
            continue

        # CHECK-NEXT
        m = re.match(r'//\s*CHECK-NEXT:\s*(.+)', stripped)
        if m:
            directives.append(CheckDirective(
                kind="CHECK-NEXT", pattern=m.group(1).strip(),
                line_num=line_num,
            ))
            continue

        # CHECK (基础)
        m = re.match(r'//\s*CHECK:\s*(.+)', stripped)
        if m:
            directives.append(CheckDirective(
                kind="CHECK", pattern=m.group(1).strip(),
                line_num=line_num,
            ))
            continue

        # 非指令行 → IR body
        ir_lines.append(line)

    ir_body = "\n".join(ir_lines)
    return run_pass, ir_body, directives


# =====================================================================
# Pass 执行
# =====================================================================

# 支持的 pass 名 → 执行函数
_PASS_REGISTRY = {
    "attention_fusion": run_attention_fusion_pass,
    "online_softmax": run_online_softmax_pass,
    "pipeline_a": build_attention_optimization_pipeline,
    "pipeline_b": build_online_softmax_pipeline,
}


def run_pass_on_ir(pass_name: str, ir_body: str) -> str:
    """
    在 IR 文本上执行指定 pass，返回变换后的 IR 文本。

    Args:
        pass_name: pass 名称（见 _PASS_REGISTRY）
        ir_body: MLIR IR 文本

    Returns:
        变换后的 IR 文本
    """
    if pass_name not in _PASS_REGISTRY:
        raise ValueError(
            f"Unknown pass: '{pass_name}'. "
            f"Available: {list(_PASS_REGISTRY.keys())}"
        )

    ctx = ir.Context()
    torch_dialect.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True

    module = ir.Module.parse(ir_body, context=ctx)
    pass_fn = _PASS_REGISTRY[pass_name]
    pass_fn(module)

    return module.operation.get_asm()


# =====================================================================
# 验证
# =====================================================================

def verify_directives(ir_output: str,
                      directives: List[CheckDirective]) -> List[CheckResult]:
    """
    验证变换后的 IR 输出满足所有 CHECK 指令。

    Returns:
        CheckResult 列表
    """
    results: List[CheckResult] = []
    ir_lines = ir_output.split("\n")
    last_match_line_idx: Optional[int] = None  # 上一个 CHECK 匹配的行索引

    for directive in directives:
        if directive.kind == "CHECK":
            found = False
            for i, line in enumerate(ir_lines):
                if directive.pattern in line:
                    found = True
                    last_match_line_idx = i
                    break
            results.append(CheckResult(
                directive=directive,
                passed=found,
                message="" if found else f"Pattern '{directive.pattern}' not found in IR output",
            ))

        elif directive.kind == "CHECK-NOT":
            found = directive.pattern in ir_output
            results.append(CheckResult(
                directive=directive,
                passed=not found,
                message="" if not found else f"Pattern '{directive.pattern}' should NOT appear but was found",
            ))

        elif directive.kind == "CHECK-SAME":
            if last_match_line_idx is None:
                results.append(CheckResult(
                    directive=directive,
                    passed=False,
                    message="CHECK-SAME without preceding CHECK match",
                ))
            else:
                line = ir_lines[last_match_line_idx]
                found = directive.pattern in line
                results.append(CheckResult(
                    directive=directive,
                    passed=found,
                    message="" if found else
                        f"Pattern '{directive.pattern}' not found on same line "
                        f"(line {last_match_line_idx + 1}): {line.strip()}",
                ))

        elif directive.kind == "CHECK-NEXT":
            if last_match_line_idx is None:
                results.append(CheckResult(
                    directive=directive,
                    passed=False,
                    message="CHECK-NEXT without preceding CHECK match",
                ))
            else:
                next_idx = last_match_line_idx + 1
                if next_idx < len(ir_lines):
                    found = directive.pattern in ir_lines[next_idx]
                    if found:
                        last_match_line_idx = next_idx
                    results.append(CheckResult(
                        directive=directive,
                        passed=found,
                        message="" if found else
                            f"Pattern '{directive.pattern}' not found on next line "
                            f"(line {next_idx + 1}): {ir_lines[next_idx].strip()}",
                    ))
                else:
                    results.append(CheckResult(
                        directive=directive,
                        passed=False,
                        message="CHECK-NEXT: no next line exists",
                    ))

        elif directive.kind == "CHECK-COUNT":
            actual = ir_output.count(directive.pattern)
            passed = actual == directive.count
            results.append(CheckResult(
                directive=directive,
                passed=passed,
                message="" if passed else
                    f"Pattern '{directive.pattern}' expected {directive.count} times, "
                    f"found {actual} times",
            ))

    return results


# =====================================================================
# 主入口
# =====================================================================

def run_filecheck(mlir_file: str, *, verbose: bool = True) -> bool:
    """
    运行一个 .mlir FileCheck 测试文件。

    Args:
        mlir_file: .mlir 文件路径
        verbose: 是否打印详细输出

    Returns:
        True 如果所有指令通过，False 否则
    """
    path = Path(mlir_file)
    if not path.exists():
        print(f"ERROR: File not found: {mlir_file}")
        return False

    mlir_text = path.read_text(encoding="utf-8")

    # 解析
    run_pass, ir_body, directives = parse_directives(mlir_text)

    if not run_pass:
        print(f"ERROR: No // RUN: directive in {mlir_file}")
        return False

    if not directives:
        print(f"WARNING: No CHECK directives in {mlir_file}")
        return True

    if verbose:
        print(f"FileCheck: {path.name}")
        print(f"  Pass: {run_pass}")
        print(f"  Directives: {len(directives)}")

    # 执行 pass
    try:
        ir_output = run_pass_on_ir(run_pass, ir_body)
    except Exception as e:
        print(f"  ERROR running pass '{run_pass}': {e}")
        return False

    # 验证
    results = verify_directives(ir_output, directives)
    all_passed = all(r.passed for r in results)

    if verbose:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            d = r.directive
            print(f"  [{status}] {d.kind}: {d.pattern}")
            if not r.passed and r.message:
                print(f"         {r.message}")
        print(f"  {'ALL PASSED' if all_passed else 'FAILED'}")

    return all_passed


def run_all_filecheck(directory: str = None, *, verbose: bool = True) -> bool:
    """
    运行目录下所有 .mlir FileCheck 测试。

    Returns:
        True 如果全部通过
    """
    if directory is None:
        directory = str(Path(__file__).parent)

    mlir_files = sorted(Path(directory).glob("*.mlir"))
    if not mlir_files:
        print(f"No .mlir files found in {directory}")
        return True

    print(f"Running {len(mlir_files)} FileCheck test(s)...")
    print("=" * 60)

    all_passed = True
    for f in mlir_files:
        passed = run_filecheck(str(f), verbose=verbose)
        if not passed:
            all_passed = False
        print()

    print("=" * 60)
    print(f"{'ALL PASSED' if all_passed else 'SOME FAILED'}: "
          f"{sum(1 for f in mlir_files if run_filecheck(str(f), verbose=False))}"
          f"/{len(mlir_files)} files")
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 运行指定文件
        success = run_filecheck(sys.argv[1])
    else:
        # 运行当前目录下所有 .mlir 文件
        success = run_all_filecheck()

    sys.exit(0 if success else 1)
