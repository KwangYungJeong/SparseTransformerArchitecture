def partition_columns(n_cols, n_arrays):
    sizes = [(n_cols + i) // n_arrays for i in range(n_arrays)]
    parts, start = [], 0
    for sz in sizes:
        parts.append(list(range(start, start + sz)))
        start += sz
    return parts

def test_partition_columns():
    # 테스트 케이스
    test_cases = [
        (1000, 8),
        (1024, 8),
        (1023, 8),
        (1000, 7),
        (5, 2),
        (10, 3)
    ]

    for n_cols, n_arrays in test_cases:
        print(f"\nTest: n_cols={n_cols}, n_arrays={n_arrays}")
        parts = partition_columns(n_cols, n_arrays)

        # 확인 1: 각 파티션 크기 출력
        sizes = [len(p) for p in parts]
        for i, part in enumerate(parts):
            print(f" PE[{i}] columns = {part[0]} ~ {part[-1]} (size: {len(part)})")

        # 확인 2: 총합이 맞는지
        total = sum(len(p) for p in parts)
        print(f" ✅ Total columns assigned: {total} / {n_cols} --> {'OK' if total == n_cols else '❌ Error'}")

        # 확인 3: 겹치거나 빠진 인덱스 없는지
        flat = sorted([idx for part in parts for idx in part])
        expected = list(range(n_cols))
        if flat == expected:
            print(" ✅ No gaps or overlaps")
        else:
            print(" ❌ Index mismatch detected!")

if __name__ == "__main__":
    test_partition_columns()
