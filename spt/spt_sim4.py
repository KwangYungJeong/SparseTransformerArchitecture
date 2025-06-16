#!/usr/bin/env python3
import numpy as np
import argparse
import spt_mask as sm   # 같은 폴더에 존재한다고 가정
import matplotlib.pyplot as plt

# ---------- 1. column partition ----------
def partition_columns(n_cols, n_partitions): # 1000개의 col과 8개의 PEArray(파티션)이 있다 가정
    sizes = [(n_cols + i) // n_partitions for i in range(n_partitions)] #[125, 125, ... 125] 총 8개 요소소
    parts = []
    start = 0
    for i in sizes:
        parts.append(list(range(start, start + i))) # [[0, 1, ..., 124], [125, ..., 249], ..., [875, ..., 999]]
        start += i
    return parts



# --------------- 2. Binary mask 받기 -----------------
def get_binary_mask(pattern: str, size: int, window_size: int = 32, stride: int = 32, dilation: int = 2):
    pattern = pattern.lower()
    if pattern == 'strided':
        mask = sm.create_strided_mask_step_by_step(size, window_size, stride)
    elif pattern == 'fixed':
        mask = sm.create_fixed_mask_step_by_step(size, window_size)
    elif pattern == 'normal':
        mask = sm.create_normal_mask_step_by_step(size)
    elif pattern == 'sliding':
        mask = sm.create_sliding_window_mask_step_by_step(size, window_size)
    elif pattern == 'dilated':
        mask = sm.create_dilated_sliding_window_mask_step_by_step(size, window_size, dilation)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return sm.convert_to_binary_mask(mask)
    

# ------------ 3. PE 클래스 ------------------
class PE:
    def __init__(self, pe_id: int, assigned_cols, d: int):
        self.id = pe_id
        self.cols = assigned_cols       # 예를 들어 [0..124]
        self.d = d                      # 임베딩 차원
        # 내부 더블 버퍼
        self.A_buffer = np.zeros(d, dtype=np.float32)  # Q 행렬의 row
        self.B_buffer = np.zeros(d, dtype=np.float32)  # K_T 행렬의 col
        self.out_buffer: float = 0.0  # output[row, col] 값
        self.cycles = 0                 # 실행 사이클
        self.results = {}               # {(row, col): output element}
    # ---------- 버퍼 주입 함수 (컨트롤러가 호출) ----------
    def set_A_buffer(self, q_row: np.ndarray):
        """Q 행렬의 row를 A 버퍼에 저장"""
        if len(q_row) != self.d:
            raise ValueError(f"Q row must have {self.d} elements, got {len(q_row)}")
        self.A_buffer = q_row
    
    def set_B_buffer(self, k_col: np.ndarray):
        """K_T 행렬의 col을 B 버퍼에 저장"""
        if len(k_col) != self.d:
            raise ValueError(f"K_T column must have {self.d} elements, got {len(k_col)}")
        self.B_buffer = k_col


    def run(self, row_idx: int, col_idx: int, verbose=False):
        """
        버퍼에 이미 Q의 row와 K_T의 col이 들어있다고 가정하고,
        두 버퍼의 내용을 곱해서 누적합을 계산.
        """
        acc = 0.0
        for i in range(self.d):
            acc += self.A_buffer[i] * self.B_buffer[i]
            self.cycles += 1
            if verbose:
                print(f"PE[{self.id}] MAC: A_buffer[{i}] * B_buffer[{i}] = {self.A_buffer[i]} * {self.B_buffer[i]} -> acc = {acc}")
        self.out_buffer = acc
        self.results[(row_idx, col_idx)] = acc
        if verbose:
            print(f"PE[{self.id}] output[{row_idx},{col_idx}] = {acc}")
        return acc

# --------------- 4. 메모리 ------------------
# 메모리 클래스에서 원하는 row, col에 해당하는 요소를 읽고 쓰는 기능을 구현
class Memory:
    """
    간단한 word-addressable SRAM 모델 (bandwidth 제한 포함)
    size_words: 메모리의 크기 (단어 단위) (예를 들어 1024 라면 메모리 크기가 1024 단어 * 4바이트 = 4096바이트)
    read_latency, write_latency: 읽기, 쓰기 지연 시간 (사이클 단위)
    bandwidth: 메모리 대역폭 (한 사이클에 읽거나 쓸 수 있는 단어 수(word))
    """
    def __init__(self, size_words: int, read_latency: int = 1, write_latency: int = 1, bandwidth: int = 1):
        self.size_words = size_words
        self.data = np.zeros(size_words, dtype=np.float32)
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.bandwidth = bandwidth
        self.cycles = 0
    
    def read(self, address: int, length: int = 1):
        """address: 시작주소, length: 읽을 단어 수"""
        # 가령 bw가 4이고 10개의 단어를 읽으려면 "3번"(4+4+2) 사이클 필요
        # 여기서 n_chunks는 읽어야 할 단어 수를 대역폭으로 나눈 값
        n_chunks = (length + self.bandwidth - 1) // self.bandwidth
        self.cycles += self.read_latency * n_chunks
        # 실제 데이터 읽기
        if address < 0 or address + length > self.size_words:
            raise ValueError("Address out of bounds")
        return self.data[address:address + length]
    

    def write(self, address: int, data):
        """address: 시작주소, data: 쓰고자 하는 데이터"""
        length = len(data)
        n_chunks = (length + self.bandwidth - 1) // self.bandwidth
        self.cycles += self.write_latency * n_chunks
        # 실제 데이터 쓰기
        if address < 0 or address + length > self.size_words:
            raise ValueError("Address out of bounds")
        self.data[address:address + length] = data
        return True
    
    # 단어 단위로 읽고 쓰는 함수
    def read_word(self, address: int):
        return self.read(address, 1)[0]  # 단어 단위로 읽기
    def write_word(self, address: int, value):
        return self.write(address, [value])  # 단어 단위로 쓰기

# --------------- 5. SRAM 클래스 ------------------
class SRAM:
    def __init__(self, size_words: int, read_latency: int = 1, write_latency: int = 1, bandwidth: int = 1):
        self.size_words = size_words
        self.data = np.zeros(size_words, dtype=np.float32)
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.bandwidth = bandwidth
        self.cycles = 0

    def read(self, address: int, length: int = 1):
        """address: 시작주소, length: 읽을 단어 수"""
        # 가령 bw가 4이고 10개의 단어를 읽으려면 "3번"(4+4+2) 사이클 필요
        # 여기서 n_chunks는 읽어야 할 단어 수를 대역폭으로 나눈 값
        n_chunks = (length + self.bandwidth - 1) // self.bandwidth
        self.cycles += self.read_latency * n_chunks
        # 실제 데이터 읽기
        if address < 0 or address + length > self.size_words:
            raise ValueError("Address out of bounds")
        return self.data[address:address + length]
    

    def write(self, address: int, data):
        """address: 시작주소, data: 쓰고자 하는 데이터"""
        length = len(data)
        n_chunks = (length + self.bandwidth - 1) // self.bandwidth
        self.cycles += self.write_latency * n_chunks
        # 실제 데이터 쓰기
        if address < 0 or address + length > self.size_words:
            raise ValueError("Address out of bounds")
        self.data[address:address + length] = data
        return True
    
    # 단어 단위로 읽고 쓰는 함수
    def read_word(self, address: int):
        return self.read(address, 1)[0]  # 단어 단위로 읽기
    def write_word(self, address: int, value):
        return self.write(address, [value])  # 단어 단위로 쓰기

# --------------- 6. DMA 클래스 ------------------
class DMA:
    def __init__(self, dram: Memory, sram: SRAM):
        self.dram = dram
        self.sram = sram

    def move_q_row_to_sram(self, A_base_addr, row_idx, d, sram_addr):
        q_row = self.dram.read(A_base_addr + row_idx * d, d)
        self.sram.write(sram_addr, q_row)
        return sram_addr

    def move_k_col_to_sram(self, B_base_addr, col_idx, d, sram_addr):
        k_col = self.dram.read(B_base_addr + col_idx * d, d)
        self.sram.write(sram_addr, k_col)
        return sram_addr

    def feed_to_pe_A_buffer(self, pe: PE, q_row):
        pe.set_A_buffer(q_row)
        
    def feed_to_pe_B_buffer(self, pe: PE, k_col):    
        pe.set_B_buffer(k_col)


# --------------- 7. PE Array 클래스 ------------------
class PEArray:
    def __init__(self, n_pes, n_cols, d):
        # 컬럼을 PE 개수만큼 균등 분할
        self.parts = partition_columns(n_cols, n_pes)
        self.pes = [PE(pe_id=i, assigned_cols=part, d=d) for i, part in enumerate(self.parts)]
        self.n_pes = n_pes

    def schedule(self, mask):
        """
        mask를 보고 어떤 PE가 어떤 (row, col) 연산을 해야 하는지 스케줄링.
        반환값: [(pe, row_idx, col_idx), ...]
        """
        n_rows, n_cols = mask.shape
        schedule_list = []
        for pe in self.pes:
            for col_idx in pe.cols:
                for row_idx in range(n_rows):
                    if mask[row_idx, col_idx] == 1:
                        schedule_list.append((pe, row_idx, col_idx))
        return schedule_list



# mask 시각화 함수
def show_mask(mask):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray_r', interpolation='nearest')
    plt.title("Mask Matrix Visualization")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.colorbar(label='Mask Value')
    plt.show()




# --------------- 0. 테스트 -----------------



# 기존 test_DMA_with_mask 함수에서 PE 생성 및 실행 부분만 아래처럼 교체
"""def test():
    
    #토큰의 dim=16
    #토큰의 개수는 16
    #PE 개수는 4개
    #-> output은 16*16 행렬이고 mask는 strided로 생성성
    #MEMORY는 1024 단어 크기
    #Q와 K_T는 이미 메모리에 저장되어 있다고 가정
    #Mask는 소프트웨어로 만들어져 NPU에 전달
    
    # ------1. 초기 작업------
    d = 16
    n_rows = 16
    n_cols = 16
    n_pes = 4
    # 메모리와 DMA 생성
    mem = Memory(size_words=1024, read_latency=2, write_latency=1, bandwidth=4)
    sram = SRAM(size_words=512, read_latency=1, write_latency=1, bandwidth=2)
    dma = DMA(dram=mem, sram=sram)
    # PE array 생성
    pe_array = PEArray(n_pes=n_pes, n_cols=n_cols, d=d)
    # Q와 K_T 행렬 저장
    Q = np.random.rand(n_rows, d).astype(np.float32)
    K_T = np.random.rand(d, n_cols).astype(np.float32)
    A_base_addr = 0 # Q 행렬의 시작 주소
    B_base_addr = 256 # K_T 행렬의 시작 주소
    for i in range(n_rows): # Q를 0번부터 255번까지 저장
        dma.dram.write(A_base_addr + i * d, Q[i])
    for i in range(n_cols): # K_T를 256번부터 511번까지 저장
        dma.dram.write(B_base_addr + i * d, K_T[:, i])

    # ------2. 마스크 생성------
    mask = get_binary_mask('strided', n_rows, window_size=int(np.sqrt(n_rows)), stride=int(np.sqrt(n_rows)))

    # ------3. PEArray 스케줄링------
    schedule = pe_array.schedule(mask)
    print("=== PE 스케줄링 결과 ===")
    for i, (pe, row_idx, col_idx) in enumerate(schedule):
        print(f"{i:3d}: PE[{pe.id}] -> output[{row_idx},{col_idx}]")
    print(f"총 연산 개수: {len(schedule)}")

    # ------4. Stage별 병렬 시뮬레이션------
    n_stages = len(pe_array.pes[0].cols)  # 각 PE가 담당하는 column의 개수 (동일 분할)
    total_cycles = 0
    print("\n== Stage별 병렬 시뮬레이션 ===")
    for stage in range(n_stages):
        print(f"\n[Stage {stage}]")
        pe_dram_cycles_list = []  # 각 PE별 DRAM 사이클 기록
        stage_cycles = []  # 각 stage에서 PE별 사이클 기록
        #각 PE가 이번 stage에서 맡은 col을 병렬로 처리
        for pe in pe_array.pes:
            if stage >= len(pe.cols):
                pe_dram_cycles_list.append(0)
                stage_cycles.append(0)
                continue  # PE가 맡은 col이 stage보다 많으면 건너뜀
            col_idx = pe.cols[stage] # 각 pe가 담당하는 cols에서 stage번째 idx
            # PE별 DRAM 사이클 측정 시작
            pe_dram_cycles_before = dma.dram.cycles
            k_col = dma.dram.read(B_base_addr + col_idx * d, d) # dma를 통해 k_col 
            pe.set_B_buffer(k_col)
            pe_cycles = 0 # 각 pe가 stage에서 소요한 사이클
            
            # 해당 PE가 담당하는 row_indices를 가져옴
            # schedule에서 해당 PE와 col_idx에 해당하는 row_indices를 찾음
            row_indices = [row_idx for (pe_s, row_idx, col_s) in schedule if pe_s == pe and col_s == col_idx]
            for row_idx in row_indices:
                q_row = dma.dram.read(A_base_addr + row_idx * d, d)
                pe.set_A_buffer(q_row)
                prev_cycles = pe.cycles
                pe.run(row_idx, col_idx, verbose=(pe.id == 0))
                pe_cycles += (pe.cycles - prev_cycles)
            # PE별 DRAM 사이클 측정 종료
            pe_dram_cycles = dma.dram.cycles - pe_dram_cycles_before
            pe_dram_cycles_list.append(pe_dram_cycles)
            
            print(f"PE[{pe.id}] completed {pe_cycles} cycles in Stage {stage}. DRAM cycles: {pe_dram_cycles}")
            stage_cycles.append(pe.cycles)
            
        max_stage_cycles = max(stage_cycles)
        total_cycles += max_stage_cycles
        print(f"Stage {stage} 전체 소요 사이클: {max_stage_cycles} (가장 오래 걸린 PE 기준)")
        print(f"  각 PE별 DRAM read/write 사이클: {pe_dram_cycles_list}")
    print(f"\n총 소요 사이클: {total_cycles} (모든 Stage 병렬 처리 기준)")
    print(f"총 DRAM read/write 사이클: {dma.dram.cycles}")
    print(stage_cycles)
if __name__ == "__main__":
    test()

    """

import matplotlib.pyplot as plt
import os

def plot_stage_bar_charts(stage_dram_cycles, stage_compute_cycles, mask_type, n, d, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    num_stages = len(stage_dram_cycles)
    num_pes = len(stage_dram_cycles[0])

    for stage_idx in range(num_stages):
        pes = list(range(num_pes))
        dram = stage_dram_cycles[stage_idx]
        compute = [stage_compute_cycles[stage_idx][i] for i in range(num_pes)]

        x = np.arange(num_pes)
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, compute, width, label='Compute')
        plt.bar(x + width/2, dram, width, label='DRAM')
        plt.title(f"[{mask_type.upper()}] Stage {stage_idx} Cycles per PE\nSize={n}x{n}, dim={d}")
        plt.xlabel("PE index")
        plt.ylabel("Cycles")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{mask_type}_stage{stage_idx}_bar_{n}x{n}_d{d}.png"))
        plt.close()

def plot_total_cycle_summary(total_compute_cycles, total_dram_cycles, mask_type, n, d, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.bar(['Compute', 'DRAM'], [total_compute_cycles, total_dram_cycles], color=['skyblue', 'orange'])
    plt.title(f"[{mask_type.upper()}] Total Cycle Summary\nSize={n}x{n}, dim={d}")
    plt.ylabel("Total Cycles")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{mask_type}_total_cycle_bar_{n}x{n}_d{d}.png"))
    plt.close()




def run_simulation_with_config(n: int, d: int, mask_type: str):
    print(f"\n\n========== [ MASK: {mask_type.upper()}, Size: {n}x{n}, Dim: {d}, PEs: 16 ] ==========")

    n_rows = n
    n_cols = n
    n_pes = 16

    total_words_needed = 2 * n * d + 1024  # Q, K_T, 여유 buffer
    mem = Memory(size_words=total_words_needed, read_latency=2, write_latency=1, bandwidth=4)
    sram = SRAM(size_words=2048, read_latency=1, write_latency=1, bandwidth=2)
    dma = DMA(dram=mem, sram=sram)
    pe_array = PEArray(n_pes=n_pes, n_cols=n_cols, d=d)

    Q = np.random.rand(n_rows, d).astype(np.float32)
    K_T = np.random.rand(d, n_cols).astype(np.float32)
    A_base_addr = 0
    B_base_addr = 1024

    for i in range(n_rows):
        dma.dram.write(A_base_addr + i * d, Q[i])
    for i in range(n_cols):
        dma.dram.write(B_base_addr + i * d, K_T[:, i])

    # 마스크 생성
    mask = get_binary_mask(mask_type, n_rows, window_size=int(np.sqrt(n_rows)), stride=int(np.sqrt(n_rows)))

    schedule = pe_array.schedule(mask)
    n_stages = len(pe_array.pes[0].cols)

    total_cycles = 0
    all_stage_cycles = []
    all_dram_cycles = []
    all_pe_compute_cycles = []
    for stage in range(n_stages):
        pe_dram_cycles_list = []
        stage_cycles = []
        all_pe_compute_cycles.append(stage_cycles)

        for pe in pe_array.pes:
            if stage >= len(pe.cols):
                pe_dram_cycles_list.append(0)
                stage_cycles.append(0)
                continue

            col_idx = pe.cols[stage]
            pe_dram_cycles_before = dma.dram.cycles
            k_col = dma.dram.read(B_base_addr + col_idx * d, d)
            pe.set_B_buffer(k_col)
            pe_cycles = 0

            row_indices = [row_idx for (pe_s, row_idx, col_s) in schedule if pe_s == pe and col_s == col_idx]
            for row_idx in row_indices:
                q_row = dma.dram.read(A_base_addr + row_idx * d, d)
                pe.set_A_buffer(q_row)
                prev_cycles = pe.cycles
                pe.run(row_idx, col_idx)
                pe_cycles += (pe.cycles - prev_cycles)

            pe_dram_cycles = dma.dram.cycles - pe_dram_cycles_before
            pe_dram_cycles_list.append(pe_dram_cycles)
            stage_cycles.append(pe.cycles - prev_cycles)

        max_stage_cycles = max(stage_cycles)
        total_cycles += max_stage_cycles
        all_stage_cycles.append(max_stage_cycles)
        all_dram_cycles.append(pe_dram_cycles_list)

        print(f"[Stage {stage}]")
        print(f"  PE 연산 사이클: {stage_cycles}")
        print(f"  DRAM 사이클: {pe_dram_cycles_list}")
        print(f"  Stage 최대 사이클: {max_stage_cycles}")

    print("\n[전체 요약]")
    print(f"  총 연산 사이클 (Stage 누적): {total_cycles}")
    print(f"  총 DRAM 접근 사이클: {dma.dram.cycles}")
    print(f"  Stage별 최대 연산 사이클: {all_stage_cycles}")
    print(f"  Stage별 DRAM 사이클: {all_dram_cycles}")
    print("========================================================\n")

    # 시각화 추가
    plot_stage_bar_charts(
        stage_dram_cycles=all_dram_cycles,
        stage_compute_cycles=all_pe_compute_cycles,
        mask_type=mask_type,
        n=n,
        d=d
    )

    plot_total_cycle_summary(
        total_compute_cycles=total_cycles,
        total_dram_cycles=dma.dram.cycles,
        mask_type=mask_type,
        n=n,
        d=d
    )




if __name__ == "__main__":
    sizes = [16, 32, 64]     # n x n 행렬
    dims = [16, 32, 64]      # 각 토큰의 차원

    for n in sizes:
        for d in dims:
            run_simulation_with_config(n=n, d=d, mask_type='strided')
            run_simulation_with_config(n=n, d=d, mask_type='fixed')

