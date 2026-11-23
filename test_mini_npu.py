"""Mini NPU 자동 테스트 (표준 라이브러리 unittest만 사용).

실행: python test_mini_npu.py
"""
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import main

CROSS = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class LabelTests(unittest.TestCase):
    def test_normalize_known_labels(self):
        for raw, want in [("+", "Cross"), ("cross", "Cross"), ("Cross", "Cross"), (" CROSS ", "Cross"),
                          ("x", "X"), ("X", "X")]:
            self.assertEqual(main.normalize_label(raw), want, raw)

    def test_unknown_labels_return_none(self):
        for raw in ("?", "", "PLUS", "✚", None):
            self.assertIsNone(main.normalize_label(raw), raw)

    def test_new_alias_needs_only_one_line(self):
        """라벨 확장 절차: LABEL_ALIASES에 한 줄 추가하면 모든 곳에서 인식된다."""
        with mock.patch.dict(main.LABEL_ALIASES, {"plus": "Cross"}):
            self.assertEqual(main.normalize_label("PLUS"), "Cross")
        self.assertIsNone(main.normalize_label("plus"))


class MacTests(unittest.TestCase):
    def test_document_examples(self):
        cross, x = main.Grid.from_rows(CROSS), main.Grid.from_rows(X)
        self.assertEqual(main.mac(cross, cross), 5.0)     # 십자가 x 십자가 필터
        self.assertEqual(main.mac(cross, x), 1.0)         # 십자가 x X 필터
        self.assertEqual(main.mac(x, x), 5.0)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            main.mac(main.Grid(3), main.Grid(5))

    def test_flat_matches_2d(self):
        for n in (3, 5, 13, 25):
            a, b = main.make_cross(n), main.make_x(n)
            self.assertEqual(main.mac(a, b), main.mac_flat(main.flatten(a), main.flatten(b)))

    def test_generated_patterns_are_judged_correctly(self):
        for n in (3, 5, 13, 25):
            cross, x = main.make_cross(n), main.make_x(n)
            self.assertEqual(main.judge_cross_x(main.mac(cross, cross), main.mac(cross, x)), "Cross")
            self.assertEqual(main.judge_cross_x(main.mac(x, cross), main.mac(x, x)), "X")


class TieTests(unittest.TestCase):
    def test_float_error_is_a_tie(self):
        self.assertNotEqual(0.1 + 0.2 + 0.3, 0.6)             # 단순 비교는 틀린다
        self.assertTrue(main.is_tie(0.1 + 0.2 + 0.3, 0.6))     # 허용오차 비교는 동점으로 본다

    def test_modes_use_different_words(self):
        self.assertEqual(main.judge_ab(0.9, 0.9 + 1e-16), "판정 불가")
        self.assertEqual(main.judge_cross_x(0.9, 0.9 + 1e-16), "UNDECIDED")

    def test_clear_difference_is_not_a_tie(self):
        self.assertEqual(main.judge_ab(1.0, 5.0), "B")
        self.assertEqual(main.judge_cross_x(5.0, 1.0), "Cross")


class InputTests(unittest.TestCase):
    def read(self, lines):
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=lines), redirect_stdout(out):
            grid = main.read_grid("필터 A")
        return grid, out.getvalue()

    def test_tabs_and_extra_spaces_are_accepted(self):
        grid, _ = self.read(["0\t1\t0", "  1   1   1  ", "0 1 0"])
        self.assertEqual(grid.cells, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def test_bad_lines_reprompt(self):
        grid, out = self.read(["1 2", "a b c", "", "0 1 0", "1 1 1", "0 1 0"])
        self.assertEqual(out.count(main.MSG_FORMAT.format(n=3)), 3)
        self.assertEqual(grid.n, 3)

    def test_interrupt_returns_none(self):
        grid, _ = self.read([EOFError()])
        self.assertIsNone(grid)


class Mode2Tests(unittest.TestCase):
    def run_mode2(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main.mode_json()
        return out.getvalue()

    def test_summary_counts_and_failure_types(self):
        out = self.run_mode2()
        self.assertIn("총 테스트: 9개", out)
        self.assertIn("통과: 6개", out)
        self.assertIn("실패: 3개", out)
        for line in ("- 라벨 오류: 1건", "- 동점(UNDECIDED): 1건", "- 크기 불일치: 1건"):
            self.assertIn(line, out)

    def test_failure_type_counter(self):
        counts = main.count_failure_types([("a", "라벨 오류: x"), ("b", "라벨 오류: y"), ("c", "동점(UNDECIDED): z")])
        self.assertEqual(counts, {"라벨 오류": 2, "동점(UNDECIDED)": 1})

    def test_perf_table_ops_are_n_squared(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main.print_perf_table([3, 5, 13, 25])
        rows = [l.split() for l in out.getvalue().splitlines() if l[:1].isdigit()]
        self.assertEqual([int(r[2]) for r in rows], [9, 25, 169, 625])


if __name__ == "__main__":
    unittest.main(verbosity=2)
