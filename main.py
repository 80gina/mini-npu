"""Mini NPU 시뮬레이터 (Python 3.8 이상, 표준 라이브러리만 사용).

십자가(Cross)와 X 패턴을 필터와의 MAC(Multiply-Accumulate) 연산 점수로 판별한다.
"""
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

EPSILON = 1e-9      # 두 점수의 차이가 이보다 작으면 동점으로 본다
REPEAT = 10         # 성능 측정 반복 횟수
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
MSG_FORMAT = "입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."


# ---------------------------------------------------------------- 데이터 구조
class Grid:
    """n x n 2차원 배열. 패턴과 필터를 담는다."""

    def __init__(self, n: int):
        self.n = n
        self.cells = [[0.0] * n for _ in range(n)]

    @classmethod
    def from_rows(cls, rows) -> "Grid":
        """2차원 리스트로 Grid를 만든다. 정사각형이 아니거나 숫자가 아니면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열이 아님")
        n = len(rows)
        grid = cls(n)
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != n:
                raise ValueError("정사각형(n x n)이 아님: %d행의 열 개수가 %d가 아님" % (i + 1, n))
            for j, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("숫자가 아닌 값이 있음 (%d행 %d열)" % (i + 1, j + 1))
                grid.set(i, j, value)
        return grid

    def get(self, i: int, j: int) -> float:
        """i행 j열의 값을 읽는다."""
        return self.cells[i][j]

    def set(self, i: int, j: int, value: float) -> None:
        """i행 j열에 값을 저장한다."""
        self.cells[i][j] = float(value)

    def show(self) -> str:
        return "\n".join(" ".join(("%g" % v) for v in row) for row in self.cells)


# ---------------------------------------------------------------- 핵심 연산
def normalize_label(raw) -> Optional[str]:
    """라벨을 표준 라벨(Cross, X)로 통일한다. '+', 'cross' -> Cross / 'x' -> X. 모르면 None."""
    key = str(raw).strip().lower()
    if key in ("+", "cross"):
        return "Cross"
    if key == "x":
        return "X"
    return None


def mac(pattern: Grid, filt: Grid) -> float:
    """패턴과 필터를 위치별로 곱해 모두 더한다. (반복문으로 직접 구현, 외부 라이브러리 없음)"""
    if pattern.n != filt.n:
        raise ValueError("크기 불일치: 패턴 %dx%d, 필터 %dx%d" % (pattern.n, pattern.n, filt.n, filt.n))
    total = 0.0
    for i in range(pattern.n):
        for j in range(pattern.n):
            total += pattern.get(i, j) * filt.get(i, j)
    return total


def is_tie(score_a: float, score_b: float) -> bool:
    """허용오차(epsilon) 기반 동점 판정."""
    return abs(score_a - score_b) < EPSILON


def judge_ab(score_a: float, score_b: float) -> str:
    """모드 1용: A / B / 판정 불가."""
    if is_tie(score_a, score_b):
        return "판정 불가"
    return "A" if score_a > score_b else "B"


def judge_cross_x(cross_score: float, x_score: float) -> str:
    """모드 2용: Cross / X / UNDECIDED."""
    if is_tie(cross_score, x_score):
        return "UNDECIDED"
    return "Cross" if cross_score > x_score else "X"


def section(title: str) -> None:
    print("#" + "-" * 39)
    print("# " + title)
    print("#" + "-" * 39)


# ---------------------------------------------------------------- 성능 분석
def make_cross(n: int) -> Grid:
    g = Grid(n)
    for i in range(n):
        for j in range(n):
            if i == n // 2 or j == n // 2:
                g.set(i, j, 1)
    return g


def measure_ms(n: int, repeat: int = REPEAT) -> float:
    """n x n MAC 연산을 repeat번 측정해 평균 시간(ms)을 돌려준다. (입출력 시간은 제외)"""
    pattern = make_cross(n)
    filt = make_cross(n)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, filt)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def print_perf_table(sizes) -> None:
    print("크기        평균 시간(ms)    연산 횟수(N²)    회당(µs)")
    print("-" * 56)
    for n in sizes:
        avg = measure_ms(n)
        ops = n * n
        label = "%dx%d" % (n, n)
        print("%-10s  %-14.3f  %-14d  %.3f" % (label, avg, ops, avg * 1000 / ops))


# ---------------------------------------------------------------- 모드 1: 사용자 입력 (3x3)
def read_grid(title: str, n: int = 3) -> Optional[Grid]:
    """n줄을 한 줄씩(공백 구분) 입력받는다. 형식이 틀리면 안내 후 그 줄을 다시 입력받는다."""
    print("%s (%d줄 입력, 공백 구분)" % (title, n))
    rows = []
    while len(rows) < n:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되었습니다.")
            return None
        parts = line.split()
        if len(parts) != n:                     # 열 개수 불일치 (빈 줄 포함)
            print(MSG_FORMAT.format(n=n))
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:                      # 숫자 파싱 실패
            print(MSG_FORMAT.format(n=n))
            continue
    return Grid.from_rows(rows)


def mode_user_input() -> None:
    section("[1] 필터 입력")
    filter_a = read_grid("필터 A")
    if filter_a is None:
        return
    print()
    filter_b = read_grid("필터 B")
    if filter_b is None:
        return
    print("\n저장 확인: 필터 A, 필터 B (각 3x3)를 저장했습니다.")
    print("[필터 A]\n" + filter_a.show() + "\n[필터 B]\n" + filter_b.show())

    section("[2] 패턴 입력")
    pattern = read_grid("패턴")
    if pattern is None:
        return

    section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    times = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        mac(pattern, filter_a)
        times.append((time.perf_counter() - start) * 1000)
    print("A 점수: %r" % score_a)
    print("B 점수: %r" % score_b)
    print("연산 시간(평균/%d회): %.3f ms" % (REPEAT, sum(times) / len(times)))
    verdict = judge_ab(score_a, score_b)
    if verdict == "판정 불가":
        print("판정: 판정 불가 (|A-B| < %g)" % EPSILON)
    else:
        print("판정: " + verdict)

    section("[4] 성능 분석 (3x3)")
    print_perf_table([3])


# ---------------------------------------------------------------- 모드 2: data.json 분석
def load_filters(raw_filters) -> Tuple[Dict[int, Dict[str, Grid]], Dict[int, str]]:
    """filters(size_N)를 읽어 라벨을 표준화한다. (사용 가능한 필터, 읽기 실패 사유)"""
    good: Dict[int, Dict[str, Grid]] = {}
    bad: Dict[int, str] = {}
    for key, value in raw_filters.items():
        m = re.match(r"^size_(\d+)$", str(key))
        if not m:
            continue
        n = int(m.group(1))
        try:
            labeled = {}
            for label_key, rows in value.items():
                label = normalize_label(label_key)
                if label is None:
                    raise ValueError("알 수 없는 필터 키: %r" % (label_key,))
                grid = Grid.from_rows(rows)
                if grid.n != n:
                    raise ValueError("필터 크기 %d가 %s와 다름" % (grid.n, key))
                labeled[label] = grid
            if "Cross" not in labeled or "X" not in labeled:
                raise ValueError("Cross, X 필터가 모두 필요함")
            good[n] = labeled
            print("✓ %s 필터 로드 완료 (Cross, X)" % key)
        except (ValueError, AttributeError) as e:
            bad[n] = "필터 오류: %s" % e
            print("✗ %s 필터 로드 실패 (%s)" % (key, e))
    return good, bad


def sort_key(name: str):
    m = re.match(r"^size_(\d+)_(.+)$", name)
    if not m:
        return (10 ** 9, name)
    idx = m.group(2)
    return (int(m.group(1)), (0, int(idx), "") if idx.isdigit() else (1, 0, idx))


def analyze_case(name: str, case, filters, bad_filters) -> Tuple[str, str]:
    """패턴 한 건을 판정한다. (PASS 또는 FAIL, 사유). 어떤 오류도 이 함수 밖으로 새지 않는다."""
    try:
        m = re.match(r"^size_(\d+)_(.+)$", name)
        if not m:
            return "FAIL", "스키마 오류: 키가 size_{N}_{idx} 형식이 아님"
        n = int(m.group(1))
        if n in bad_filters:
            return "FAIL", bad_filters[n]
        if n not in filters:
            return "FAIL", "스키마 오류: size_%d 필터가 data.json에 없음" % n
        if not isinstance(case, dict) or "input" not in case or "expected" not in case:
            return "FAIL", "스키마 오류: input 또는 expected가 없음"
        expected = normalize_label(case["expected"])
        if expected is None:
            return "FAIL", "라벨 오류: expected %r는 Cross(+)나 X(x)가 아님" % (case["expected"],)
        pattern = Grid.from_rows(case["input"])
        if pattern.n != n:
            return "FAIL", "크기 불일치: 패턴 %dx%d, 필터 %dx%d" % (pattern.n, pattern.n, n, n)
    except ValueError as e:
        return "FAIL", "스키마 오류: %s" % e

    cross_score = mac(pattern, filters[n]["Cross"])
    x_score = mac(pattern, filters[n]["X"])
    verdict = judge_cross_x(cross_score, x_score)
    print("Cross 점수: %r" % cross_score)
    print("X 점수: %r" % x_score)
    if verdict == expected:
        print("판정: %s | expected: %s | PASS" % (verdict, expected))
        return "PASS", ""
    if verdict == "UNDECIDED":
        reason = "동점(UNDECIDED): |Cross-X| = %.3g < %g, 동점 규칙에 따라 FAIL" % (abs(cross_score - x_score), EPSILON)
    else:
        reason = "판정 불일치: 판정 %s, expected %s" % (verdict, expected)
    print("판정: %s | expected: %s | FAIL (%s)" % (verdict, expected, reason.split(":")[0]))
    return "FAIL", reason


def mode_json() -> None:
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        raw_filters = data["filters"]
        raw_patterns = data["patterns"]
        if not isinstance(raw_filters, dict) or not isinstance(raw_patterns, dict):
            raise TypeError("filters/patterns 형식이 객체가 아님")
    except (OSError, ValueError, KeyError, TypeError) as e:
        print("data.json을 읽을 수 없습니다: %s" % e)
        return

    section("[1] 필터 로드")
    filters, bad_filters = load_filters(raw_filters)

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = []
    for name in sorted(raw_patterns, key=sort_key):
        print("--- %s ---" % name)
        status, reason = analyze_case(name, raw_patterns[name], filters, bad_filters)
        if status == "FAIL" and not reason.startswith(("동점", "판정 불일치")):
            print("FAIL (%s)" % reason)
        results.append((name, status, reason))

    section("[3] 성능 분석 (평균/%d회)" % REPEAT)
    print_perf_table([3, 5, 13, 25])

    section("[4] 결과 요약")
    total = len(results)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    print("총 테스트: %d개" % total)
    print("통과: %d개" % passed)
    print("실패: %d개" % (total - passed))
    failures = [(name, reason) for name, status, reason in results if status == "FAIL"]
    if failures:
        print("\n실패 케이스:")
        for name, reason in failures:
            print("- %s: %s" % (name, reason))
        print("\n(상세 원인 분석 및 복잡도 설명은 README.md의 \"결과 리포트\" 섹션 참고)")


# ---------------------------------------------------------------- 실행
def main() -> None:
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]\n")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    while True:
        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되어 종료합니다.")
            return
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력하세요.")
    print()
    if choice == "1":
        mode_user_input()
    else:
        mode_json()


if __name__ == "__main__":
    main()
