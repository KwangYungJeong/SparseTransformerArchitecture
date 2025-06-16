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
