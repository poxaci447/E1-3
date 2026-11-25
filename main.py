import json
import math
import re
import time
from pathlib import Path
from typing import Any

EPSILON = 1e-9
BENCHMARK_REPEATS = 10


class Matrix2D:
    """N×N 크기의 2차원 패턴 및 필터를 저장하고 원소에 접근하는 데이터 구조입니다."""

    def __init__(self, data: list[list[float]]) -> None:
        if not data or not isinstance(data, list):
            raise ValueError("비어 있지 않은 2차원 리스트가 필요합니다.")
        size = len(data)
        for row in data:
            if not isinstance(row, list) or len(row) != size:
                raise ValueError(f"{size}x{size} 정사각 행렬 형식이 아닙니다.")
        self.size = size
        self._data: list[list[float]] = [
            [float(val) for val in row] for row in data
        ]

    def get(self, row: int, col: int) -> float:
        """지정된 (row, col) 위치의 값을 반환합니다."""
        return self._data[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        """지정된 (row, col) 위치에 값을 저장합니다."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("행렬 원소는 유한한 숫자여야 합니다.")
        self._data[row][col] = float(value)

    def display(self) -> None:
        """행렬 내용을 보기 좋게 출력합니다."""
        for row in self._data:
            formatted = " ".join(f"{v:g}" for v in row)
            print(f"  {formatted}")

    @classmethod
    def from_raw(cls, raw: Any, expected_size: int | None = None) -> "Matrix2D":
        """JSON 등 외부 2차원 배열 데이터를 검증해 Matrix2D 인스턴스로 변환합니다."""
        if not isinstance(raw, list) or not raw:
            raise ValueError("2차원 배열 형식이 아닙니다.")

        row_count = len(raw)
        if expected_size is not None and row_count != expected_size:
            raise ValueError(f"행 크기 불일치 (기대: {expected_size}, 실제: {row_count})")

        target_size = expected_size if expected_size is not None else row_count
        parsed: list[list[float]] = []

        for r_idx, row in enumerate(raw):
            if not isinstance(row, list):
                raise ValueError(f"{r_idx + 1}번째 행이 리스트 형식이 아닙니다.")
            if len(row) != target_size:
                raise ValueError(
                    f"{r_idx + 1}번째 행의 열 크기 불일치 (기대: {target_size}, 실제: {len(row)})"
                )
            parsed_row: list[float] = []
            for c_idx, val in enumerate(row):
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"({r_idx}, {c_idx}) 위치에 유효하지 않은 숫자 값이 있습니다.")
                parsed_row.append(float(val))
            parsed.append(parsed_row)

        return cls(parsed)


def normalize_filter_key(key: Any) -> str:
    """필터 키를 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(key, str):
        raise ValueError(f"필터 키가 문자열이 아닙니다: {key!r}")
    cleaned = key.strip().lower()
    if cleaned == "cross":
        return "Cross"
    if cleaned == "x":
        return "X"
    raise ValueError(f"지원하지 않는 필터 키입니다: {key!r}")


def normalize_expected_label(label: Any) -> str:
    """expected 값을 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(label, str):
        raise ValueError(f"expected 값이 문자열이 아닙니다: {label!r}")
    cleaned = label.strip().lower()
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    raise ValueError(f"지원하지 않는 expected 라벨입니다: {label!r}")


def extract_size_from_key(key: str, pattern_mode: bool = False) -> int:
    """'size_N' 또는 'size_N_idx' 형태의 키에서 행렬 크기 N을 추출합니다."""
    regex = r"size_(\d+)_(\d+)" if pattern_mode else r"size_(\d+)"
    match = re.fullmatch(regex, key)
    if not match:
        expected_fmt = "size_{N}_{idx}" if pattern_mode else "size_{N}"
        raise ValueError(f"키 형식 오류 (기대 형식: {expected_fmt}, 실제: {key!r})")
    size = int(match.group(1))
    if size <= 0:
        raise ValueError(f"크기 N은 양의 정수여야 합니다: {size}")
    return size


def compute_mac(pattern: Matrix2D, kernel: Matrix2D) -> float:
    """외부 라이브러리 없이 이중 반복문으로 MAC(Multiply-Accumulate) 연산을 수행합니다."""
    if pattern.size != kernel.size:
        raise ValueError(
            f"패턴({pattern.size}x{pattern.size})과 필터({kernel.size}x{kernel.size})의 크기가 일치하지 않습니다."
        )

    total = 0.0
    n = pattern.size
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * kernel.get(r, c)
    return total


def judge_scores(
    score_a: float,
    score_b: float,
    label_a: str,
    label_b: str,
    undecided_label: str = "UNDECIDED",
    epsilon: float = EPSILON,
) -> str:
    """허용오차(epsilon) 정책을 적용해 두 필터의 MAC 점수를 비교·판정합니다."""
    if abs(score_a - score_b) < epsilon:
        return undecided_label
    return label_a if score_a > score_b else label_b


def benchmark_mac(
    pattern: Matrix2D,
    kernel: Matrix2D,
    repeats: int = BENCHMARK_REPEATS,
) -> tuple[float, int]:
    """I/O를 제외하고 MAC 연산 함수 호출 구간만 repeats회 반복 측정해 (평균 ms, N²)을 반환합니다."""
    start = time.perf_counter()
    for _ in range(repeats):
        compute_mac(pattern, kernel)
    elapsed_sec = time.perf_counter() - start
    avg_ms = (elapsed_sec / repeats) * 1000.0
    ops_count = pattern.size * pattern.size
    return avg_ms, ops_count


def format_score(score: float, high_precision: bool = False) -> str:
    """부동소수점 근접 비교 시 정밀 자릿수를 드러내도록 점수를 포맷합니다."""
    if high_precision:
        return f"{score:.16f}"
    return f"{score}"


def input_matrix_3x3(title: str) -> Matrix2D:
    """3×3 행렬을 한 줄씩(공백 구분) 입력받고 행/열 개수 및 숫자 파싱을 검증합니다."""
    size = 3
    print(f"\n{title} ({size}줄 입력, 공백 구분)")
    rows: list[list[float]] = []

    while len(rows) < size:
        line_num = len(rows) + 1
        raw = input(f"  {line_num}행: ").strip()
        tokens = raw.split()

        if len(tokens) != size:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
            continue

        try:
            values = [float(tok) for tok in tokens]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("비유한 숫자")
        except ValueError:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
            continue

        rows.append(values)

    return Matrix2D(rows)


def print_performance_table(records: list[tuple[int, float, int]], repeats: int = BENCHMARK_REPEATS) -> None:
    """크기(N×N), 평균 시간(ms), 연산 횟수(N²) 성능 표를 출력합니다."""
    print(f"\n#----------------------------------------")
    print(f"# 성능 분석 (평균/{repeats}회)")
    print(f"#----------------------------------------")
    print(f"{'크기':<10} {'평균 시간(ms)':>14} {'연산 횟수':>10}")
    print("-" * 38)
    for size, avg_ms, ops in records:
        size_label = f"{size}x{size}"
        print(f"{size_label:<10} {avg_ms:>14.4f} {ops:>10}")


def run_user_input_mode() -> None:
    """모드 1: 사용자 입력(3×3) 기반 MAC 시뮬레이션을 수행합니다."""
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a = input_matrix_3x3("필터 A")
    print("✓ 필터 A 저장 확인:")
    filter_a.display()

    filter_b = input_matrix_3x3("필터 B")
    print("✓ 필터 B 저장 확인:")
    filter_b.display()

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern = input_matrix_3x3("패턴")
    print("✓ 패턴 저장 확인:")
    pattern.display()

    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    avg_ms, ops = benchmark_mac(pattern, filter_a, repeats=BENCHMARK_REPEATS)

    is_close = abs(score_a - score_b) < EPSILON
    result = judge_scores(score_a, score_b, label_a="A", label_b="B", undecided_label="판정 불가")

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    print(f"A 점수: {format_score(score_a, high_precision=is_close)}")
    print(f"B 점수: {format_score(score_b, high_precision=is_close)}")
    print(f"연산 시간(평균/{BENCHMARK_REPEATS}회): {avg_ms:.4f} ms")
    if result == "판정 불가":
        print(f"판정: 판정 불가 (|A - B| < {EPSILON:g})")
    else:
        print(f"판정: {result}")

    print_performance_table([(3, avg_ms, ops)], repeats=BENCHMARK_REPEATS)


def run_json_mode(json_path: Path) -> None:
    """모드 2: data.json을 로드·검증하여 일괄 판정 및 크기별 성능 분석을 수행합니다."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"[오류] 데이터 파일을 찾을 수 없습니다: {json_path}")
        return
    except (OSError, json.JSONDecodeError) as e:
        print(f"[오류] data.json을 읽거나 파싱하는 중 오류가 발생했습니다: {e}")
        return

    if not isinstance(raw_data, dict):
        print("[오류] data.json 최상위 구조는 객체(dict)여야 합니다.")
        return

    raw_filters = raw_data.get("filters")
    raw_patterns = raw_data.get("patterns")
    if not isinstance(raw_filters, dict) or not isinstance(raw_patterns, dict):
        print("[오류] data.json에 유효한 'filters' 및 'patterns' 섹션이 없습니다.")
        return

    print("\n#----------------------------------------")
    print("# [1] 필터 로드")
    print("#----------------------------------------")
    filters_by_size: dict[int, dict[str, Matrix2D]] = {}

    for size_key, filter_group in raw_filters.items():
        try:
            size_n = extract_size_from_key(size_key, pattern_mode=False)
            if not isinstance(filter_group, dict):
                raise ValueError("필터 그룹은 객체(dict)여야 합니다.")

            normalized_group: dict[str, Matrix2D] = {}
            for raw_label, raw_matrix in filter_group.items():
                std_label = normalize_filter_key(raw_label)
                normalized_group[std_label] = Matrix2D.from_raw(raw_matrix, expected_size=size_n)

            filters_by_size[size_n] = normalized_group
            loaded_labels = ", ".join(normalized_group.keys())
            print(f"✓ {size_key} 필터 로드 완료 ({loaded_labels})")
        except ValueError as e:
            print(f"✗ {size_key} 필터 로드 실패: {e}")

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#----------------------------------------")

    total_cases = 0
    pass_count = 0
    fail_cases: list[tuple[str, str]] = []

    # 3x3 성능 분석 포함을 위한 기본 3x3 기준 데이터
    default_3x3_cross = Matrix2D([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    benchmark_targets: dict[int, tuple[Matrix2D, Matrix2D]] = {
        3: (default_3x3_cross, default_3x3_cross)
    }

    for case_id, item in raw_patterns.items():
        total_cases += 1
        print(f"\n--- {case_id} ---")

        if not isinstance(item, dict):
            reason = "스키마 오류: 패턴 항목이 객체(dict)가 아닙니다."
            print(f"판정: ERROR | FAIL ({reason})")
            fail_cases.append((case_id, reason))
            continue

        try:
            size_n = extract_size_from_key(case_id, pattern_mode=True)
        except ValueError as e:
            reason = f"키 형식 오류: {e}"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_cases.append((case_id, reason))
            continue

        filter_pair = filters_by_size.get(size_n)
        if not filter_pair or "Cross" not in filter_pair or "X" not in filter_pair:
            reason = f"필터 누락: size_{size_n}에 대한 Cross/X 필터가 존재하지 않습니다."
            print(f"판정: ERROR | FAIL ({reason})")
            fail_cases.append((case_id, reason))
            continue

        try:
            expected = normalize_expected_label(item.get("expected"))
        except ValueError as e:
            reason = f"라벨 정규화 오류: {e}"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_cases.append((case_id, reason))
            continue

        try:
            pattern_matrix = Matrix2D.from_raw(item.get("input"), expected_size=size_n)
        except ValueError as e:
            reason = f"크기/데이터 불일치: {e}"
            print(f"판정: ERROR | expected: {expected} | FAIL ({reason})")
            fail_cases.append((case_id, reason))
            continue

        cross_filter = filter_pair["Cross"]
        x_filter = filter_pair["X"]

        if size_n not in benchmark_targets:
            benchmark_targets[size_n] = (pattern_matrix, cross_filter)

        score_cross = compute_mac(pattern_matrix, cross_filter)
        score_x = compute_mac(pattern_matrix, x_filter)
        is_close = abs(score_cross - score_x) < EPSILON
        predicted = judge_scores(score_cross, score_x, label_a="Cross", label_b="X", undecided_label="UNDECIDED")

        print(f"Cross 점수: {format_score(score_cross, high_precision=is_close)}")
        print(f"X 점수: {format_score(score_x, high_precision=is_close)}")

        if predicted == expected:
            pass_count += 1
            print(f"판정: {predicted} | expected: {expected} | PASS")
        else:
            if predicted == "UNDECIDED":
                reason = f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross - X| < {EPSILON:g})"
            else:
                reason = f"판정 불일치 (판정: {predicted}, expected: {expected})"
            print(f"판정: {predicted} | expected: {expected} | FAIL ({reason})")
            fail_cases.append((case_id, reason))

    # 로드된 필터 중 패턴이 모두 실패한 크기가 있더라도 성능 표에 포함
    for size_n, filter_pair in filters_by_size.items():
        if size_n not in benchmark_targets and "Cross" in filter_pair:
            benchmark_targets[size_n] = (filter_pair["Cross"], filter_pair["Cross"])

    perf_records: list[tuple[int, float, int]] = []
    for size_n in sorted(benchmark_targets.keys()):
        pat_mat, ker_mat = benchmark_targets[size_n]
        avg_ms, ops = benchmark_mac(pat_mat, ker_mat, repeats=BENCHMARK_REPEATS)
        perf_records.append((size_n, avg_ms, ops))

    print_performance_table(perf_records, repeats=BENCHMARK_REPEATS)

    fail_count = len(fail_cases)
    print("\n#----------------------------------------")
    print("# [4] 결과 요약")
    print("#----------------------------------------")
    print(f"총 테스트: {total_cases}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {fail_count}개")
    if fail_cases:
        print("실패 케이스:")
        for cid, reason in fail_cases:
            print(f"  - {cid}: {reason}")
    else:
        print("실패 케이스 없음 (라벨 정규화 및 epsilon 동점 처리 정책 적용)")


def main() -> None:
    """Mini NPU Simulator 실행 진입점입니다."""
    data_path = Path(__file__).resolve().parent / "data.json"

    print("=== Mini NPU Simulator ===")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")

    try:
        while True:
            choice = input("선택: ").strip()
            if choice == "1":
                run_user_input_mode()
                break
            if choice == "2":
                run_json_mode(data_path)
                break
            print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
    except (KeyboardInterrupt, EOFError):
        print("\n[안내] 종료 신호가 감지되어 프로그램을 안전하게 종료합니다.")


if __name__ == "__main__":
    main()