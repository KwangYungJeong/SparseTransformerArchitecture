#!/usr/bin/env python3
"""
spt_sim.py
==========

Sparse-Transformer PE-array simulator.

- Column-wise tiling:  N×N attention 출력 행렬의 열을 `num_pe` 개의 PE array가 나눠 맡음
- 각 PE array: multiplier 1개 + accumulator 1개  →  d-차원 dot product = d cycles
- dot product는  Q[row] · Kᵀ[:, col]  (마스크 bit==1인 경우에만 계산)
- 사이클 계산
      • parallel_cycles   = max(PE_i.cycles)  (모든 PE array가 동시에 동작한다고 가정)
      • sequential_cycles = sum(PE_i.cycles) (한 array씩 돌린다면)

CLI 옵션
---------
--size           시퀀스 길이 N (= 행/열 수)                    [default 1024]
--d              임베딩 차원 d                                [default 64]
--num_pe         PE array 개수                                [default 8]
--pattern        마스크 패턴(strided|fixed|normal|sliding|dilated)
--window_size    패턴용 로컬 윈도 크기                        [default 32]
--stride         strided 패턴의 stride                         [default 32]
--dilation       dilated sliding window의 dilation             [default 2]
--seed           난수 시드                                     [default 0]
--save_outputs   연산 결과를 .npy 로 저장할 경로 (선택)

사용 예
-------
$ python spt_sim.py --size 1000 --d 128 --num_pe 8 --pattern strided --window_size 64 --stride 64
"""
import argparse
import numpy as np
import spt_mask as sm   # 같은 디렉터리에 두 파일이 있다고 가정


# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #
def partition_columns(n_cols: int, n_arrays: int):
    """
    n_cols 개의 열을 n_arrays 개의 그룹으로 고르게 분할
    반환값: [ [col_idx, ...], [col_idx, ...], ... ]
    """
    sizes = [(n_cols + i) // n_arrays for i in range(n_arrays)]  # 앞쪽에 1개씩 더 배분
    parts, start = [], 0
    for sz in sizes:
        parts.append(list(range(start, start + sz)))
        start += sz
    return parts


# --------------------------------------------------------------------------- #
#  PE-array 클래스
# --------------------------------------------------------------------------- #
class PEArray:
    """
    하나의 PE array 시뮬레이터 (multiplier 1 + accumulator 1)
    """
    def __init__(self, array_id: int, assigned_cols, d: int):
        self.id = array_id
        self.cols = assigned_cols
        self.d = d
        self.cycles = 0
        self.results = {}          # {(row, col): value}

    def run(self, Q: np.ndarray, K_T: np.ndarray, mask: np.ndarray):
        """
        Q : (N, d)   - 각 row 가 query 벡터
        K_T : (d, N) - Kᵀ  (열마다 key 벡터)
        mask : (N, N) binary
        """
        for col in self.cols:
            k_vec = K_T[:, col]           # (d,)
            rows = np.nonzero(mask[:, col])[0]   # 마스크 bit==1 인 행들
            for row in rows:
                self.results[(row, col)] = float(np.dot(Q[row], k_vec))
                self.cycles += self.d     # d 사이클(1 MAC/사이클 가정)

        return self.results, self.cycles


# --------------------------------------------------------------------------- #
#  전체 시뮬레이션
# --------------------------------------------------------------------------- #
def simulate(Q, K, mask, num_pe):
    """
    Q, K : (N, d)  (K 는 전치해서 사용)
    mask  : (N, N) binary
    num_pe: PE array 개수
    """
    N, d = Q.shape
    K_T = K.T               # (d, N)
    col_parts = partition_columns(N, num_pe)

    arrays = [PEArray(i, cols, d) for i, cols in enumerate(col_parts)]

    # 각 PE array 실행
    out = np.zeros((N, N), dtype=np.float32)
    cycles_each = []
    for pe in arrays:
        results, cyc = pe.run(Q, K_T, mask)
        cycles_each.append(cyc)
        for (r, c), v in results.items():
            out[r, c] = v

    parallel_cycles = max(cycles_each)        # 동시에 구동
    sequential_cycles = sum(cycles_each)      # 직렬 구동

    return out, cycles_each, parallel_cycles, sequential_cycles


# --------------------------------------------------------------------------- #
#  CLI 진입점
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Sparse-Transformer PE-array simulator")
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--num_pe', type=int, default=8)
    parser.add_argument('--pattern',
                        choices=['strided', 'fixed', 'normal', 'sliding', 'dilated'],
                        default='strided')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_outputs')
    args = parser.parse_args()

    np.random.seed(args.seed)

    N, d = args.size, args.d
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)

    # --------------------------- 마스크 생성 --------------------------- #
    if args.pattern == 'strided':
        mask = sm.create_strided_mask_step_by_step(N, args.window_size, args.stride)
    elif args.pattern == 'fixed':
        mask = sm.create_fixed_mask_step_by_step(N, args.window_size)
    elif args.pattern == 'normal':
        mask = sm.create_normal_mask_step_by_step(N)
    elif args.pattern == 'sliding':
        mask = sm.create_sliding_window_mask_step_by_step(N, args.window_size)
    elif args.pattern == 'dilated':
        mask = sm.create_dilated_sliding_window_mask_step_by_step(
            N, args.window_size, args.dilation)
    else:
        raise ValueError("Unknown pattern")

    mask_bin = sm.convert_to_binary_mask(mask)

    # --------------------------- 시뮬레이션 --------------------------- #
    out, cycles_list, par_cyc, seq_cyc = simulate(Q, K, mask_bin, args.num_pe)

    print(f"\n=== Simulation finished ({args.num_pe} PE arrays) ===")
    for i, c in enumerate(cycles_list):
        print(f"PE[{i}] cycles : {c}")
    print(f"Total cycles (parallel)  : {par_cyc}")
    print(f"Total cycles (sequential): {seq_cyc}")
    computed = np.count_nonzero(out)
    print(f"Computed elements        : {computed}/{N*N} "
          f"({computed/(N*N):.2%} of full attention)")

    if args.save_outputs:
        np.save(args.save_outputs, out)
        print(f"Saved output matrix → {args.save_outputs}")


if __name__ == "__main__":
    main()
