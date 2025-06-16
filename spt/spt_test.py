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
"""def test_PE():
    n_cols = 1000
    n_partitions = 8
    d = 64

    parts = partition_columns(n_cols, n_partitions)
    PE_array = [PE(pe_id=i, assigned_cols=part, d=d) for i, part in enumerate(parts)]

    Q = np.random.rand(n_cols, d)
    K_T = np.random.rand(d, n_cols)
    mask = get_binary_mask('strided', n_cols)


    total_cycles = 0
    for pe in PE_array:
        results, cycles = pe.run(Q, K_T, mask, verbose=(pe.id == 0))
        total_cycles += cycles
        print(f"PE[{pe.id}] completed with {cycles} cycles.")

    print(f"\nTotal cycles across all PE: {total_cycles}")

    # 마스크 시각화
    show_mask(mask)"""

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

# DMA 테스트(+ 마스크에 따른 PE 실행)
"""def test_DMA_with_mask():
    print("=== DMA + Mask 테스트 ===")
    d = 16
    n_rows = 16
    n_cols = 16
    mem = Memory(size_words=1024, read_latency=2, write_latency=1, bandwidth=4)
    dma = DMA(memory=mem)

    # Q 행렬의 row를 메모리에 쓰기
    A_base_addr = 0
    Q = np.random.rand(n_rows, d).astype(np.float32)
    for i in range(n_rows):
        dma.memory.write(A_base_addr + i * d, Q[i])

    # K_T 행렬의 col을 메모리에 쓰기
    B_base_addr = 256
    K_T = np.random.rand(d, n_cols).astype(np.float32)
    for i in range(n_cols):
        dma.memory.write(B_base_addr + i * d, K_T[:, i])

    # 마스크 생성 (예시: 대각선만 1)
    mask = get_binary_mask('strided', n_rows, window_size = int(np.sqrt(n_rows)), stride = int(np.sqrt(n_rows)))
    print("Mask:\n", mask)

    # PE 생성 및 실행
    pe = PE(pe_id=0, assigned_cols=list(range(n_cols)), d=d)
    cycles_dict = {}
    for col_idx in range(n_cols):
        # K_T 행렬의 col을 DMA를 통해 로드
        k_col = dma.load_B_col(B_base_addr, col_idx, d)
        dma.feed_to_pe_B_buffer(pe, k_col)
        for row_idx in range(n_rows):
            if mask[row_idx, col_idx] == 1:
                # 해당 Q 행렬의 row를 DMA를 통해 로드
                q_row = dma.load_A_row(A_base_addr, row_idx, d)
                dma.feed_to_pe_A_buffer(pe, q_row)
                prev_cycles = pe.cycles
                pe.run(row_idx, col_idx)
                cycles_for_this_output = pe.cycles - prev_cycles
                print(f"output[{row_idx},{col_idx}] 계산에 걸린 사이클: {cycles_for_this_output}")
                cycles_dict[(row_idx, col_idx)] = cycles_for_this_output
    print(f"PE[{pe.id}] results: {pe.results}")


if __name__ == "__main__":
    test_DMA_with_mask()"""

# 기존 test_DMA_with_mask 함수에서 PE 생성 및 실행 부분만 아래처럼 교체
"""
def test_DMA_with_mask():
    print("=== DMA + Mask + PEArray 스케줄러 테스트 ===")
    d = 16
    n_rows = 16
    n_cols = 16
    n_pes = 4
    mem = Memory(size_words=1024, read_latency=2, write_latency=1, bandwidth=4)
    dma = DMA(memory=mem)

    # Q, K_T 메모리에 저장
    A_base_addr = 0
    Q = np.random.rand(n_rows, d).astype(np.float32)
    for i in range(n_rows):
        dma.memory.write(A_base_addr + i * d, Q[i])
    B_base_addr = 256
    K_T = np.random.rand(d, n_cols).astype(np.float32)
    for i in range(n_cols):
        dma.memory.write(B_base_addr + i * d, K_T[:, i])

    # 마스크 생성
    mask = get_binary_mask('strided', n_rows, window_size=int(np.sqrt(n_rows)), stride=int(np.sqrt(n_rows)))
    print("Mask:\n", mask)

    # PEArray 생성 및 스케줄링
    pe_array = PEArray(n_pes=n_pes, n_cols=n_cols, d=d)
    schedule = pe_array.schedule(mask)

    # 스케줄에 따라 DMA가 데이터 공급, PE가 연산
    for pe, row_idx, col_idx in schedule:
        k_col = dma.load_B_col(B_base_addr, col_idx, d)
        dma.feed_to_pe_B_buffer(pe, k_col)
        q_row = dma.load_A_row(A_base_addr, row_idx, d)
        dma.feed_to_pe_A_buffer(pe, q_row)
        prev_cycles = pe.cycles
        pe.run(row_idx, col_idx)
        cycles_for_this_output = pe.cycles - prev_cycles
        print(f"PE[{pe.id}] output[{row_idx},{col_idx}] 계산에 걸린 사이클: {cycles_for_this_output}")

    # 결과 출력
    for pe in pe_array.pes:
        print(f"PE[{pe.id}] results: {pe.results}")

if __name__ == "__main__":
    test_DMA_with_mask()
"""