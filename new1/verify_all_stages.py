#!/usr/bin/env python3
"""
verify_all_stages.py — UL-UNAS Frame 0 逐函数完整验证
=====================================================
Reads C per-stage output .bin files and MATLAB golden .bin files,
generates complete per-bin verification tables for every pipeline stage.

Usage:
    1. First run the C binary to dump per-stage outputs:
       ./ulunas_nr_full --dump ../stft/test_wavs/noisy_fileid_1.wav
       (Creates dump_c/frame0_*.bin)

    2. MATLAB golden should be in dump_matlab/frame0_*.bin

    3. Run:
       python3 verify_all_stages.py dump_matlab/ dump_c/

Output format (per stage):
    整体 SNR: XX.XX dB
    ┌─────┬──────────┬──────────┬──────────┬──────────┬───────────┐
    │ Bin │ C Q20    │ MATLAB Q20│ 误差(LSB)│ 误差(%)  │ 有效位宽   │
    ├─────┼──────────┼──────────┼──────────┼──────────┼───────────┤
    │  0  │  202965  │  202965  │     0    │  0.0000  │    38     │
    ...
    └─────┴──────────┴──────────┴──────────┴──────────┴───────────┘
"""

import sys
import os
import struct
import math

# ─── Stage definitions in Main_infer.m pipeline order ───
STAGES = [
    # (display_name, golden_file, c_file, n_elements, description)
    ("STFT_real",    "frame0_stft_real.bin",   "frame0_stft_real_c.bin",   257, "STFT 实部"),
    ("STFT_imag",    "frame0_stft_imag.bin",   "frame0_stft_imag_c.bin",   257, "STFT 虚部"),
    ("log_gen",      "frame0_log_gen.bin",     "frame0_log_gen_c.bin",     257, "Log-Magnitude 压缩"),
    ("BM",           "frame0_bm.bin",          "frame0_bm_c.bin",          129, "ERB 频带合并"),
    ("E0_XConv",     "frame0_e0_xconv.bin",    "frame0_e0_xconv_c.bin",    780, "Encoder Layer 0 (XConv)"),
    ("E1_XMB0",      "frame0_e1_xmb0.bin",     "frame0_e1_xmb0_c.bin",     792, "Encoder Layer 1 (XMB0)"),
    ("E2_XDWS0",     "frame0_e2_xdws0.bin",    "frame0_e2_xdws0_c.bin",    792, "Encoder Layer 2 (XDWS0)"),
    ("E3_XMB1",      "frame0_e3_xmb1.bin",     "frame0_e3_xmb1_c.bin",    1056, "Encoder Layer 3 (XMB1)"),
    ("E4_XDWS1",     "frame0_e4_xdws1.bin",    "frame0_e4_xdws1_c.bin",    528, "Encoder Layer 4 (XDWS1)"),
    ("GDPRNN1",      "frame0_gdprnn1.bin",     "frame0_gdprnn1_c.bin",     528, "GDPRNN Block 0"),
    ("GDPRNN2",      "frame0_gdprnn2.bin",     "frame0_gdprnn2_c.bin",     528, "GDPRNN Block 1"),
    ("Decoder",      "frame0_decoder.bin",      "frame0_decoder_c.bin",     129, "Decoder 输出"),
    ("Sigmoid",      "frame0_sigmoid.bin",      "frame0_sigmoid_c.bin",     129, "Sigmoid 激活"),
    ("BS",           "frame0_bs.bin",           "frame0_bs_c.bin",          257, "ERB 频带展开"),
    ("MASK_real",    "frame0_mask_real.bin",    "frame0_mask_real_c.bin",   257, "CRM Mask 实部"),
    ("MASK_imag",    "frame0_mask_imag.bin",    "frame0_mask_imag_c.bin",   257, "CRM Mask 虚部"),
]

def read_bin_int32(path, n):
    with open(path, 'rb') as f:
        data = f.read(n * 4)
    return list(struct.unpack(f'<{n}i', data))

def snr_db(c_data, m_data):
    """SNR = 10*log10(signal_power / noise_power)"""
    sig = sum(m * m for m in m_data)
    noise = sum((c - m) ** 2 for c, m in zip(c_data, m_data))
    if noise == 0:
        return 999.0
    return 10.0 * math.log10(sig / noise)

def effective_bitwidth(val_q20):
    """Effective bit-width = integer bits + 20 fractional bits"""
    if val_q20 == 0:
        return 20
    int_bits = int(math.floor(math.log2(abs(val_q20)))) + 1
    return max(20, int_bits + 20)

def fmt_err_pct(err, golden):
    """Error percentage with 4 decimal places"""
    if golden == 0:
        return 0.0 if err == 0 else float('inf')
    return abs(err / golden) * 100.0

def verify_stage(name, c_file, golden_file, n_elem, c_dir, g_dir, desc):
    """Generate verification table for one stage."""
    c_path = os.path.join(c_dir, c_file)
    g_path = os.path.join(g_dir, golden_file)

    has_c = os.path.exists(c_path)
    has_g = os.path.exists(g_path)

    print(f"\n{'='*80}")
    print(f"  {name} — {desc}")
    print(f"{'='*80}")

    if not has_g:
        print(f"  [SKIP] Golden file not found: {g_path}")
        return None
    if not has_c:
        print(f"  [SKIP] C output file not found: {c_path}")
        # Still print golden summary
        golden = read_bin_int32(g_path, n_elem)
        print(f"  Golden available: {n_elem} elements")
        print(f"  C output NOT available — run C binary with --dump first")
        return None

    c_data = read_bin_int32(c_path, n_elem)
    g_data = read_bin_int32(g_path, n_elem)

    snr = snr_db(c_data, g_data)
    max_err = max(abs(c - g) for c, g in zip(c_data, g_data))
    mean_err = sum(abs(c - g) for c, g in zip(c_data, g_data)) / n_elem

    # Check bit-exact range
    bit_exact_count = sum(1 for c, g in zip(c_data, g_data) if c == g)
    bit_exact_pct = bit_exact_count / n_elem * 100

    print(f"\n  ### 数值对比 (Q20 整数 vs MATLAB golden {golden_file})")
    print(f"  ```")
    print(f"  整体 SNR: {snr:.2f} dB")
    print(f"  最大误差: {max_err} LSB")
    print(f"  平均误差: {mean_err:.2f} LSB")
    print(f"  Bit-exact bins: {bit_exact_count}/{n_elem} ({bit_exact_pct:.1f}%)")

    if bit_exact_count == n_elem:
        print(f"  所有 bin bit-exact (0 LSB error)")
    elif bit_exact_pct > 99.9:
        print(f"  接近 bit-exact ({bit_exact_pct:.2f}% bins 零误差)")

    # Select representative bins for the table: 0, 4, 8, 16, 32, 48, 64, and then every 8th
    total_bins = n_elem
    if total_bins <= 16:
        bin_indices = list(range(total_bins))
    else:
        # Smart selection: first few bins + evenly spaced samples
        bin_indices = []
        step = max(1, total_bins // 16)
        for i in range(0, total_bins, step):
            bin_indices.append(i)
        # Ensure last bin is included
        if bin_indices[-1] != total_bins - 1:
            bin_indices.append(total_bins - 1)
        # Add key bins
        for key in [0, 4, 8, 16, 32, 48, 64]:
            if key < total_bins and key not in bin_indices:
                bin_indices.append(key)
        bin_indices.sort()
        # Trim to max 20 rows
        if len(bin_indices) > 20:
            bin_indices = bin_indices[:20]

    # Print table
    print(f"  ```")
    print(f"  ┌{'─'*5}┬{'─'*12}┬{'─'*12}┬{'─'*10}┬{'─'*10}┬{'─'*11}┐")
    print(f"  │ {'Bin':>4}│ {'C Q20':>10} │ {'MATLAB Q20':>10} │ {'误差(LSB)':>8} │ {'误差(%)':>8} │ {'有效位宽':>9} │")
    print(f"  ├{'─'*5}┼{'─'*12}┼{'─'*12}┼{'─'*10}┼{'─'*10}┼{'─'*11}┤")

    for b in bin_indices:
        c_val = c_data[b]
        g_val = g_data[b]
        err = c_val - g_val
        pct = fmt_err_pct(err, g_val)
        ebits = effective_bitwidth(c_val)
        pct_str = f"{pct:.4f}" if pct != float('inf') else "∞"
        print(f"  │ {b:>4}│ {c_val:>10} │ {g_val:>10} │ {err:>8} │ {pct_str:>8} │ {ebits:>9} │")

    print(f"  └{'─'*5}┴{'─'*12}┴{'─'*12}┴{'─'*10}┴{'─'*10}┴{'─'*11}┘")
    print(f"  ```")

    return {'name': name, 'snr': snr, 'max_err': max_err, 'mean_err': mean_err,
            'bit_exact_pct': bit_exact_pct, 'n_elem': n_elem}

def main():
    if len(sys.argv) < 3:
        g_dir = "dump_matlab"
        c_dir = "dump_c"
        print(f"Usage: {sys.argv[0]} <golden_dir> <c_output_dir>")
        print(f"Using defaults: golden={g_dir}, c_output={c_dir}")
    else:
        g_dir = sys.argv[1]
        c_dir = sys.argv[2]

    print("=" * 80)
    print("  UL-UNAS Frame 0 — 逐函数 C Fixed-Point vs MATLAB Golden 完整验证")
    print("=" * 80)
    print(f"  Golden dir: {g_dir}")
    print(f"  C output dir: {c_dir}")
    print()

    # Check golden availability
    available_golden = 0
    for name, gf, cf, n, desc in STAGES:
        if os.path.exists(os.path.join(g_dir, gf)):
            available_golden += 1

    print(f"  Golden files available: {available_golden}/{len(STAGES)}")
    print()

    results = []
    for name, gf, cf, n, desc in STAGES:
        r = verify_stage(name, cf, gf, n, c_dir, g_dir, desc)
        if r:
            results.append(r)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  汇总")
    print(f"{'='*80}")
    print(f"  ┌{'─'*16}┬{'─'*6}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*12}┐")
    print(f"  │ {'Stage':<14} │ {'Bins':>4} │ {'SNR(dB)':>8} │ {'MaxErr':>8} │ {'MeanErr':>8} │ {'BitExact%':>10} │")
    print(f"  ├{'─'*16}┼{'─'*6}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*12}┤")
    for r in results:
        print(f"  │ {r['name']:<14} │ {r['n_elem']:>4} │ {r['snr']:>8.2f} │ {r['max_err']:>8} │ {r['mean_err']:>8.2f} │ {r['bit_exact_pct']:>9.1f}% │")
    print(f"  └{'─'*16}┴{'─'*6}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*12}┘")

    # Save results
    print(f"\n  验证完成。{len(results)} 个 stage 已对比。")


if __name__ == '__main__':
    main()
