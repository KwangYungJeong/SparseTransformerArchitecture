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


    def run(self, Q: np.ndarray, K_T: np.ndarray, mask: np.ndarray, verbose=False):        #ndarray가 파이썬의 list로 안보고 벡터처럼 연산산
        """Q: (N,d) 행렬, K_T: (d,N) 행렬, mask: (N,N) 행렬(output의 형태를 나타냄)"""
        for col in self.cols:
            k_col = K_T[:, col]
            # mask가 0인 경우는 연산하지 않음, 1인 경우만 연산
            rows = np.where(mask[:, col] == 1)[0]
            for row in rows:
                q_row = Q[row, :]
                acc = 0.0
                #if verbose:
                #    print(f"PE[{self.id}] processing Q[{row},:] and K_T[:,{col}]")
                # output[row, col] 요소의 연산 Q의 row와 K_T col element wise 곱셈. MAC 하나 당 한 사이클
                for i in range(self.d):
                    acc += q_row[i] * k_col[i]
                    self.cycles += 1
                #    if verbose:
                #        print(f"PE[{self.id}] MAC: Q[{row},{i}] * K_T[{i},{col}] = {q_row[i]} * {k_col[i]} -> acc = {acc}")
                if verbose:
                    print(f"PE[{self.id}] output[{row},{col}] = {acc}")
                self.results[(row,col)] = acc # output[row, col] 값 넣기
        return self.results, self.cycles

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


# --------------- 5. Simulation ------------------






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
"""def test_partition_columns():
    test_cases = [
        (1000, 8),
        (1024, 8),
        (1023, 8),
        (5, 2),
        (10, 3)
    ]
    for n_cols, n_partitions in test_cases:
        print(f"\nTest: n_cols = {n_cols}, n_partitions = {n_partitions}")
        parts = partition_columns(n_cols, n_partitions)

        # 확인1 - 각 PEArray가 담당할 output 열 번호 출력
        for i, part in enumerate(parts):
            print(f"\n PEArray[{i}]는 output의 다음 인덱스의 column들을 가집니다.")
            print(f"\n PEArray[{i}] = {part[0]} ~ {part[-1]} (size = {len(part)})")
"""

# test_PE 함수 내에서 mask 생성 후 호출 예시:
def test_PE():
    n_cols = 16
    n_partitions = 4
    d = 16

    parts = partition_columns(n_cols, n_partitions)
    PE_array = [PE(pe_id=i, assigned_cols=part, d=d) for i, part in enumerate(parts)]

    Q = np.random.rand(n_cols, d)
    K_T = np.random.rand(d, n_cols)
    mask = get_binary_mask('strided', n_cols, window_size=int(np.sqrt(n_cols)), stride=int(np.sqrt(n_cols)))


    total_cycles = 0
    for pe in PE_array:
        results, cycles = pe.run(Q, K_T, mask, verbose=(pe.id == 0))
        total_cycles += cycles
        print(f"PE[{pe.id}] completed with {cycles} cycles.")

    print(f"\nTotal cycles across all PE: {total_cycles}")

    # 마스크 시각화
    show_mask(mask)

# Memory 클래스 테스트
"""def test_Memory():
    print("=== Memory 클래스 테스트 ===")
    mem = Memory(size_words=16, read_latency=2, write_latency=1, bandwidth=4)
    print("초기 메모리:", mem.data)
    print("\n누적 사이클:", mem.cycles)
    

    # 1. Write test
    print("\n[Write] 0~7번 주소에 데이터 쓰기")
    mem.write(0, np.arange(8, dtype=np.float32))
    print("메모리 상태:", mem.data)
    print("\n누적 사이클:", mem.cycles)
    # BW가 4이므로 한 사이클에 4개 단어를 쓸 수 있고 메모리에는 첫 사이클에 0~3이 0~3번 주소에 쓰이고
    # 두번째 사이클에 4~7이 4~7번 주소에 쓰인다. 따라서 총 2 사이클이 소요된다. cycle=2

    # 2. Read test
    print("\n[Read] 0~7번 주소 데이터 읽기")
    data = mem.read(0, 8)
    print("읽은 데이터:", data)
    print("\n누적 사이클:", mem.cycles)
    # 0~7번 주소 있는 데이터 읽고 4 단어 읽을 때 2 사이클 소요된다. 총 8개이므로 4 사이클 소요된다. 누적 cycle=6

    # 3. 단일 word write/read
    print("\n[Write/Read Word] 10번 주소에 123.45 쓰고 읽기")
    mem.write_word(10, 123.45)
    print("\n누적 사이클:", mem.cycles)
    # 10번 주소에 123.45 쓰는 것이고 BW가 4이기 때문에 한 단어 쓰는데 1 사이클 소요(BW4>1)
    # 누적 사이클은 7
    val = mem.read_word(10)
    print("\n한 번 읽은 이후 누적 사이클:", mem.cycles)
    # 10번 주소에서 하나 읽어왔으므로 2 사이클 소요
    # 누적 사이클은 9
    print("10번 주소 값:", val)

    # 5. 예외 처리 테스트 (범위 밖 접근)
    print("\n[예외 테스트] 범위 밖 읽기/쓰기")
    try:
        mem.read(14, 5)
    except ValueError as e:
        print("읽기 예외 발생:", e)
    try:
        mem.write(15, [1,2,3])
    except ValueError as e:
        print("쓰기 예외 발생:", e)"""

if __name__ == "__main__":
    test_PE()