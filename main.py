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
        print("아직 준비 중인 모드입니다.")


if __name__ == "__main__":
    main()
