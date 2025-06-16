#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from spt_row import PEArray, DMA, Memory, SRAM, get_binary_mask   # ← row 버전 import

# ---------------------------------------------------------------------------
# 1. Row-wise 시뮬레이터
# ---------------------------------------------------------------------------
def simulate(n_rows, n_cols, d, n_pes, mask_pattern):
    # ---------- 메모리 / DMA ----------
    total_words_needed = n_rows * d + n_cols * d + 1024
    mem  = Memory(size_words=total_words_needed, read_latency=2, write_latency=1, bandwidth=4)
    sram = SRAM(size_words=2048, read_latency=1, write_latency=1, bandwidth=2)
    dma  = DMA(dram=mem, sram=sram)
    pe_array = PEArray(n_pes=n_pes, n_rows=n_rows, d=d)   # ← row 버전

    # ---------- Q, K_T ----------
    Q   = np.random.rand(n_rows, d).astype(np.float32)
    K_T = np.random.rand(d, n_cols).astype(np.float32)
    A_base = 0
    B_base = n_rows * d
    for i in range(n_rows):
        dma.dram.write(A_base + i * d, Q[i])
    for i in range(n_cols):
        dma.dram.write(B_base + i * d, K_T[:, i])

    # ---------- Mask / Schedule ----------
    mask = get_binary_mask(mask_pattern, n_rows,
                           window_size=int(np.sqrt(n_rows)),
                           stride=int(np.sqrt(n_rows)))
    schedule = pe_array.schedule(mask)
    n_stages = len(pe_array.pes[0].rows)          # ← row 수가 stage 수

    # ---------- 누적 변수 ----------
    total_cycles = total_dram_cycles = total_pe_cycles = 0
    pe_row_cnt        = np.zeros(n_pes)
    per_pe_active_cyc = np.zeros(n_pes)
    max_ops_per_stage = []

    # ---------- Stage 루프 ----------
    for stage in range(n_stages):
        stage_pe_cycles, stage_dram_cycles, stage_ops = [], [], []

        for pe in pe_array.pes:
            if stage >= len(pe.rows):          # 이 PE가 맡은 row보다 stage가 크면 skip
                stage_pe_cycles.append(0)
                stage_dram_cycles.append(0)
                stage_ops.append(0)
                continue

            row_idx = pe.rows[stage]
            dram_before_pe = dma.dram.cycles

            # ---- Q row (row_idx) DRAM read  : row-wise 이므로 stage당 1회 ----
            q_row = dma.dram.read(A_base + row_idx * d, d)
            pe.set_A_buffer(q_row)

            pe_cycles_this_stage = 0
            # ---- Column loop ----
            col_indices = [c for (pe_s, r, c) in schedule
                           if pe_s is pe and r == row_idx]
            for col_idx in col_indices:
                k_col = dma.dram.read(B_base + col_idx * d, d)
                pe.set_B_buffer(k_col)
                prev = pe.cycles
                pe.run(row_idx, col_idx)
                pe_cycles_this_stage += (pe.cycles - prev)

            pe_dram_cycles = dma.dram.cycles - dram_before_pe

            # ---- 기록 ----
            stage_pe_cycles.append(pe_cycles_this_stage)
            stage_dram_cycles.append(pe_dram_cycles)
            stage_ops.append(len(col_indices))

            pe_row_cnt[pe.id]        += 1                      # row 하나 처리
            per_pe_active_cyc[pe.id] += pe_cycles_this_stage

        # ---------- Stage 요약 ----------
        stage_pe_max   = max(stage_pe_cycles)
        stage_dram_max = max(stage_dram_cycles)
        stage_ops_max  = max(stage_ops)

        total_pe_cycles   += stage_pe_max
        total_dram_cycles += stage_dram_max
        total_cycles      += (stage_pe_max + stage_dram_max)   # DRAM+PE 합
        max_ops_per_stage.append(stage_ops_max)

    # ---------- Utilization ----------
    util_op = pe_row_cnt.sum() / sum(max_ops_per_stage) if max_ops_per_stage else 0
    util_cycle_overall = per_pe_active_cyc.sum() / (total_cycles * n_pes) if total_cycles else 0
    util_cycle_per_pe  = per_pe_active_cyc / total_cycles if total_cycles else np.zeros(n_pes)

    return {
        "total_cycles"      : total_cycles,
        "total_dram_cycles" : total_dram_cycles,
        "total_pe_cycles"   : total_pe_cycles,
        "util_op"           : util_op,
        "util_cycle_overall": util_cycle_overall,
        "util_cycle_per_pe" : util_cycle_per_pe,
    }

# ---------------------------------------------------------------------------
# 2. 그래프 함수 (column-버전과 동일)
# ---------------------------------------------------------------------------
def plot_pe_cycle_util(per_pe_util, title, filename=None):
    plt.figure(figsize=(6,4))
    plt.bar(range(len(per_pe_util)), per_pe_util)
    plt.ylim(0,1)
    plt.xlabel("PE index")
    plt.ylabel("Cycle-based Utilization")
    plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_cycle_comp_per_mask(results, sizes, prefix="cycle_comp_row"):
    bar_w, x = 0.35, np.arange(len(sizes))
    for mask in ["strided", "fixed"]:
        plt.clf()
        dram = [r["total_dram_cycles"] for r in results[mask]]
        pe   = [r["total_pe_cycles"]   for r in results[mask]]
        tot  = [r["total_cycles"]      for r in results[mask]]

        plt.bar(x - bar_w/2, dram, width=bar_w, label="DRAM")
        plt.bar(x + bar_w/2, pe,   width=bar_w, label="PE")
        plt.plot(x, tot, 'k^-', lw=2, label="Total")

        plt.xticks(x, sizes)
        plt.xlabel("Output Size (n_rows = n_cols)")
        plt.ylabel("Cycles")
        plt.title(f"[Row-wise] {mask.capitalize()} – DRAM / PE / TOTAL")
        plt.grid(True, axis='y')
        plt.legend()

        plt.savefig(f"{prefix}_{mask}.png", bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------
# 3. 실행 & 결과 -------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d       = 256
    n_pes   = 16
    sizes   = [64, 128, 256, 512, 1024]
    results = {"strided": [], "fixed": []}

    for n in sizes:
        for mask in ["strided", "fixed"]:
            res = simulate(n, n, d, n_pes, mask)
            results[mask].append(res)

            util_per_pe = res["util_cycle_per_pe"]
            plot_pe_cycle_util(util_per_pe,
                               f"[Row] {mask.capitalize()} – Util (n={n})",
                               filename=f"row_pe_util_{mask}_{n}.png")

    plot_cycle_comp_per_mask(results, sizes)
