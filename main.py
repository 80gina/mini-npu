"""Mini NPU 시뮬레이터 (Python 3.8 이상, 표준 라이브러리만 사용).

십자가(Cross)와 X 패턴을 필터와의 MAC(Multiply-Accumulate) 연산 점수로 판별한다.
"""
import json
import os
import platform
import re
import time
from typing import Dict, List, Optional, Tuple

EPSILON = 1e-9      # 두 점수의 차이가 이보다 작으면 동점으로 본다
REPEAT = 10         # 성능 측정 반복 횟수
# 기본은 main.py 옆의 data.json(과제 제공 데이터). 환경 변수 MINI_NPU_DATA로 다른 파일을 지정할 수 있다.
DATA_FILE = os.environ.get("MINI_NPU_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data.json")
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
# 표준 라벨로 바꿀 수 있는 표기 목록 (소문자 기준). 새 표기를 지원하려면 여기에 한 줄만 추가한다.
LABEL_ALIASES = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}


def normalize_label(raw) -> Optional[str]:
    """라벨을 표준 라벨(Cross, X)로 통일한다. 앞뒤 공백과 대소문자는 무시한다. 모르면 None.

    라벨 규칙을 이 함수 한 곳에 모아 두었기 때문에 필터 키, expected, 출력 비교가
    모두 같은 기준을 쓰고, 새 표기를 추가해도 고칠 곳이 LABEL_ALIASES 하나뿐이다.
    """
    return LABEL_ALIASES.get(str(raw).strip().lower())


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
    """n x n MAC 연산을 repeat번 측정해 평균 시간(ms)을 돌려준다. (입출력 시간은 제외)

    첫 호출은 캐시·바이트코드 준비 때문에 느릴 수 있어서, 측정 전에 한 번 돌려 버린다(워밍업).
    """
    pattern = make_cross(n)
    filt = make_cross(n)
    mac(pattern, filt)      # 워밍업 1회 (시간에 포함하지 않음)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, filt)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def env_line() -> str:
    """측정 환경 한 줄: 같은 값이라도 컴퓨터가 다르면 시간이 달라지므로 함께 기록한다."""
    return "측정 환경: Python %s, %s, 반복 %d회(워밍업 1회 제외)" % (
        platform.python_version(), platform.system() + " " + platform.machine(), REPEAT)


def print_perf_table(sizes) -> None:
    print(env_line())
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


def failure_type(reason: str) -> str:
    """실패 사유 문장에서 유형(라벨 오류, 크기 불일치, 동점 등)을 뽑는다."""
    for kind in ("라벨 오류", "크기 불일치", "동점", "스키마 오류", "필터 오류", "판정 불일치"):
        if reason.startswith(kind):
            return "동점(UNDECIDED)" if kind == "동점" else kind
    return "기타"


def count_failure_types(failures) -> Dict[str, int]:
    """[(케이스, 사유), ...]를 유형별 개수로 센다. (등장 순서 유지)"""
    counts: Dict[str, int] = {}
    for _name, reason in failures:
        kind = failure_type(reason)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# 실패 유형 -> 한 줄 요약에 쓰는 표준 키
SUMMARY_KEYS = {
    "라벨 오류": "fail_label",
    "크기 불일치": "fail_size",
    "동점(UNDECIDED)": "fail_tie",
    "스키마 오류": "fail_schema",
    "필터 오류": "fail_schema",
    "판정 불일치": "fail_logic",
}


def summary_line(total: int, passed: int, failures, bad_filters) -> str:
    """기계가 읽기 쉬운 한 줄 요약. 예: 요약: total=9 pass=6 fail=3 fail_label=1 ..."""
    counts = {"fail_label": 0, "fail_size": 0, "fail_tie": 0, "fail_schema": 0, "fail_logic": 0}
    for kind, count in count_failure_types(failures).items():
        counts[SUMMARY_KEYS.get(kind, "fail_schema")] += count
    parts = ["total=%d" % total, "pass=%d" % passed, "fail=%d" % (total - passed)]
    parts += ["%s=%d" % (key, value) for key, value in counts.items()]
    parts.append("failed_filters=%d" % len(bad_filters))
    return "요약: " + " ".join(parts)


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
    print(summary_line(total, passed, failures, bad_filters))
    if bad_filters:
        print("로드 실패 필터: " + ", ".join("size_%d" % n for n in sorted(bad_filters)))
    if failures:
        print("\n실패 유형별 개수:")
        for kind, count in count_failure_types(failures).items():
            print("- %s: %d건" % (kind, count))
        print("\n실패 케이스:")
        for name, reason in failures:
            print("- %s: %s" % (name, reason))
        print("\n(상세 원인 분석 및 복잡도 설명은 README.md의 \"결과 리포트\" 섹션 참고)")


# ---------------------------------------------------------------- 보너스: 패턴 생성기 + 1차원 최적화
def make_x(n: int) -> Grid:
    """N x N X 패턴: 두 대각선을 1로 채운다."""
    g = Grid(n)
    for i in range(n):
        g.set(i, i, 1)
        g.set(i, n - 1 - i, 1)
    return g


def flatten(grid: Grid) -> List[float]:
    """2차원 배열을 길이 N²의 1차원 리스트로 편다. (행 우선)"""
    return [grid.cells[i][j] for i in range(grid.n) for j in range(grid.n)]


def mac_flat(pattern: List[float], filt: List[float]) -> float:
    """1차원 배열 버전 MAC. 칸마다 get() 호출 없이 인덱스 하나로 접근한다."""
    if len(pattern) != len(filt):
        raise ValueError("크기 불일치: 길이 %d, %d" % (len(pattern), len(filt)))
    total = 0.0
    for k in range(len(pattern)):
        total += pattern[k] * filt[k]
    return total


def avg_ms(func, repeat: int = REPEAT) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def print_compare_table(sizes) -> None:
    """같은 입력, 같은 반복 횟수로 2차원 MAC과 1차원 MAC을 비교한다. (펴는 시간은 제외)"""
    print("크기        2D 평균(ms)    1D 평균(ms)    속도 향상    점수 일치")
    print("-" * 64)
    for n in sizes:
        cross, x = make_cross(n), make_x(n)
        cf, xf = flatten(cross), flatten(x)
        t2 = avg_ms(lambda: mac(cross, x))
        t1 = avg_ms(lambda: mac_flat(cf, xf))
        same = abs(mac(cross, x) - mac_flat(cf, xf)) < EPSILON
        speed = ("%.2f배" % (t2 / t1)) if t1 > 0 else "-"
        print("%-10s  %-13.3f  %-13.3f  %-10s  %s" % ("%dx%d" % (n, n), t2, t1, speed, "일치" if same else "불일치"))


def mode_bonus() -> None:
    try:
        raw = input("패턴 크기 N 입력 (3~25): ").strip()
        while not (raw.isdigit() and 3 <= int(raw) <= 25):
            print("입력 형식 오류: 3 이상 25 이하의 정수를 입력하세요.")
            raw = input("패턴 크기 N 입력 (3~25): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되었습니다.")
        return
    n = int(raw)
    cross, x = make_cross(n), make_x(n)
    print()
    section("[1] 생성된 패턴 (%dx%d)" % (n, n))
    print("Cross:\n" + cross.show() + "\n")
    print("X:\n" + x.show() + "\n")
    section("[2] 생성 패턴 판정 (생성된 Cross/X를 필터로 사용)")
    for label, pattern in (("Cross", cross), ("X", x)):
        cs, xs = mac(pattern, cross), mac(pattern, x)
        print("%s 패턴 -> Cross 점수: %s, X 점수: %s, 판정: %s" % (label, cs, xs, judge_cross_x(cs, xs)))
    print()
    section("[3] 성능 분석 (생성 패턴, 평균/%d회)" % REPEAT)
    print_perf_table([3, 5, 13, 25, n] if n not in (3, 5, 13, 25) else [3, 5, 13, 25])
    print()
    section("[4] 최적화 전/후 비교 (2차원 -> 1차원 배열)")
    print_compare_table([3, 5, 13, 25] if n in (3, 5, 13, 25) else [3, 5, 13, 25, n])


# ---------------------------------------------------------------- 실행
def main() -> None:
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]\n")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 보너스: 패턴 생성기 + 1차원 최적화 비교")
    while True:
        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되어 종료합니다.")
            return
        if choice in ("1", "2", "3"):
            break
        print("1, 2, 3 중 하나를 입력하세요.")
    print()
    if choice == "1":
        mode_user_input()
    elif choice == "2":
        mode_json()
    else:
        mode_bonus()


if __name__ == "__main__":
    main()
