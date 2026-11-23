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


class EpsilonBoundaryTests(unittest.TestCase):
    """EPSILON 경계: 차이가 정확히 1e-9이면 동점이 아니고(< 비교), 그보다 작으면 동점이다."""

    def test_just_below_is_tie(self):
        self.assertTrue(main.is_tie(1.0, 1.0 + 0.99e-9))

    def test_exactly_epsilon_is_not_tie(self):
        self.assertFalse(main.is_tie(0.0, main.EPSILON))

    def test_just_above_is_not_tie(self):
        self.assertFalse(main.is_tie(1.0, 1.0 + 1.01e-9))

    def test_symmetric(self):
        self.assertEqual(main.is_tie(0.5, 0.5 + 1e-12), main.is_tie(0.5 + 1e-12, 0.5))


class AliasSafetyTests(unittest.TestCase):
    """새 alias를 추가했을 때 규칙이 깨지지 않았는지 자동으로 확인한다."""

    def test_aliases_map_only_to_standard_labels(self):
        self.assertTrue(set(main.LABEL_ALIASES.values()) <= {"Cross", "X"})

    def test_alias_keys_are_lowercase_and_trimmed(self):
        for key in main.LABEL_ALIASES:
            self.assertEqual(key, key.strip().lower())

    def test_every_alias_round_trips(self):
        for key, label in main.LABEL_ALIASES.items():
            self.assertEqual(main.normalize_label(key), label)
            self.assertEqual(main.normalize_label(key.upper()), label)


class ReportFormatTests(unittest.TestCase):
    def test_tie_failure_has_fixed_prefix(self):
        """동점 FAIL 사유는 항상 '동점(UNDECIDED):'로 시작한다."""
        tie_filters = {3: {"Cross": main.Grid.from_rows([[0.1, 0.2, 0.3], [0, 0, 0], [0, 0, 0]]),
                           "X": main.Grid.from_rows([[0.3, 0.2, 0.1], [0, 0, 0], [0, 0, 0]])}}
        case = {"input": [[1, 1, 1], [0, 0, 0], [0, 0, 0]], "expected": "X"}
        out = io.StringIO()
        with redirect_stdout(out):
            status, reason = main.analyze_case("size_3_1", case, tie_filters, {})
        self.assertEqual(status, "FAIL")
        self.assertTrue(reason.startswith("동점(UNDECIDED):"), reason)

    def test_summary_line_uses_standard_keys(self):
        failures = [("a", "라벨 오류: x"), ("b", "크기 불일치: y"), ("c", "동점(UNDECIDED): z")]
        line = main.summary_line(9, 6, failures, {13: "필터 오류: ..."})
        for token in ("total=9", "pass=6", "fail=3", "fail_label=1", "fail_size=1", "fail_tie=1",
                      "fail_schema=0", "fail_logic=0", "failed_filters=1"):
            self.assertIn(token, line)

    def test_missing_filter_is_fail(self):
        status, reason = main.analyze_case("size_7_1", {"input": [[1]], "expected": "+"}, {}, {})
        self.assertEqual(status, "FAIL")
        self.assertIn("필터가 data.json에 없음", reason)

    def test_failed_filter_listed_in_summary(self):
        self.assertIn("failed_filters=2", main.summary_line(0, 0, [], {5: "x", 13: "y"}))

    def test_env_line_has_python_and_repeat(self):
        line = main.env_line()
        self.assertIn("Python", line)
        self.assertIn("반복 %d회" % main.REPEAT, line)


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
