"""
FileCheck 集成测试
===================

将 FileCheck .mlir 测试文件集成到 unittest 框架中，
与 Phase 1/2 测试统一运行。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from mlir.tests.filecheck.run_filecheck import run_filecheck


_FILECHECK_DIR = Path(__file__).parent / "filecheck"


class TestFileCheck(unittest.TestCase):
    """基于 .mlir 文件的 FileCheck 测试。"""

    def test_basic_fusion(self):
        """基础融合 FileCheck：scale + mask + softmax → fused op。"""
        self.assertTrue(
            run_filecheck(str(_FILECHECK_DIR / "basic_fusion.mlir"), verbose=False),
            "basic_fusion.mlir FileCheck failed",
        )

    def test_dynamic_shapes(self):
        """不同维度的 online softmax FileCheck。"""
        self.assertTrue(
            run_filecheck(str(_FILECHECK_DIR / "dynamic_shapes.mlir"), verbose=False),
            "dynamic_shapes.mlir FileCheck failed",
        )

    def test_negative_no_match(self):
        """反例 FileCheck：不满足条件时不应融合。"""
        self.assertTrue(
            run_filecheck(str(_FILECHECK_DIR / "negative_no_match.mlir"), verbose=False),
            "negative_no_match.mlir FileCheck failed",
        )


if __name__ == "__main__":
    unittest.main()
