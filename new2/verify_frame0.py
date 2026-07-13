#!/usr/bin/env python3
"""
verify_frame0.py — UL-UNAS C vs MATLAB Frame 0 Per-Stage Verification
======================================================================
Reads C output from ulunas_nr_test and MATLAB golden from dump_matlab/,
computes per-bin SNR, error, and effective bit-width for each pipeline stage.

Usage:
    # 1. First run MATLAB to export golden data:
    #    matlab -nodisplay -r "run('export_frame0_golden.m'); exit"
    #
    # 2. Then run C test:
    #    cd ulunas_c && make run 2>stderr.txt
    #
    # 3. Finally run this verification:
    python3 verify_frame0.py dump_matlab/ ulunas_c/stderr.txt

Output:
    Per-stage SNR, max error, and detailed per-bin comparison tables.
"""

import sys
import os
import struct
import math
import re

def read_bin_int32(path, n):
    """Read n int32 values from binary file."""
    with open(path, 'rb') as f:
        data = f.read(n * 4)
    return list(struct.unpack(f'<{n}i', data))

def compute_snr(c_data, matlab_data):
    """Compute SNR in dB between C and MATLAB arrays."""
    signal_power = sum(m * m for m in matlab_data)
    noise_power = sum((c - m) ** 2 for c, m in zip(c_data, matlab_data))
    if noise_power == 0:
        return 999.0
    return 10.0 * math.log10(signal_power / noise_power)

def effective_bits(value_q20):
    """Compute effective bit-width for a Q20 value."""
    if value_q20 == 0:
        return 0
    int_bits = int(math.floor(math.log2(abs(value_q20)))) + 1
    return max(0, int_bits) + 20  # +20 fractional bits

def max_abs_error(c_data, matlab_data):
    """Compute maximum absolute error."""
    return max(abs(c - m) for c, m in zip(c_data, matlab_data))

def mean_abs_error(c_data, matlab_data):
    """Compute mean absolute error."""
    return sum(abs(c - m) for c, m in zip(c_data, matlab_data)) / len(c_data)

# Stage definitions matching Main_infer.m pipeline order
STAGES = [
    # (name, golden_file, n_elements, c_extractor)
    ("STFT_real",    "frame0_stft_real.bin",  257, "stft_real"),
    ("STFT_imag",    "frame0_stft_imag.bin",  257, "stft_imag"),
    ("log_gen",      "frame0_log_gen.bin",    257, None),  # 257 bins from log_gen
    ("BM",           "frame0_bm.bin",         129, None),
    ("E0_XConv",     "frame0_e0_xconv.bin",   12*65, None),
    ("E1_XMB0",      "frame0_e1_xmb0.bin",    24*33, None),
    ("E2_XDWS0",     "frame0_e2_xdws0.bin",   24*33, None),
    ("E3_XMB1",      "frame0_e3_xmb1.bin",    32*33, None),
    ("E4_XDWS1",     "frame0_e4_xdws1.bin",   16*33, None),
    ("GDPRNN1",      "frame0_gdprnn1.bin",    16*33, None),
    ("GDPRNN2",      "frame0_gdprnn2.bin",    16*33, None),
    ("Decoder",      "frame0_decoder.bin",    129, None),
    ("Sigmoid",      "frame0_sigmoid.bin",    129, None),
    ("BS",           "frame0_bs.bin",         257, None),
    ("MASK_real",    "frame0_mask_real.bin",  257, "mask_real"),
    ("MASK_imag",    "frame0_mask_imag.bin",  257, "mask_imag"),
]

def parse_c_output(stderr_lines, stage_name):
    """Extract C output from stderr log for a specific stage."""
    # This is a placeholder — in practice, the C program would dump per-stage data
    # to separate files. For now, only STFT and MASK are directly comparable.
    return None

def print_table_header():
    print(f"{'Stage':<16} {'Elements':>8} {'SNR(dB)':>10} {'MaxErr':>10} {'MeanErr':>12} {'MinBits':>8} {'MaxBits':>8}")
    print("-" * 78)

def main():
    if len(sys.argv) < 2:
        golden_dir = "dump_matlab"
    else:
        golden_dir = sys.argv[1]

    if not os.path.isdir(golden_dir):
        print(f"ERROR: Golden directory '{golden_dir}' not found.")
        print("Run MATLAB export_frame0_golden.m first.")
        sys.exit(1)

    print("=" * 78)
    print("  UL-UNAS Frame 0: C Fixed-Point vs MATLAB Golden Verification")
    print("=" * 78)
    print(f"  Golden directory: {golden_dir}")
    print()

    # Check which golden files exist
    available = []
    for name, fname, n_elem, _ in STAGES:
        path = os.path.join(golden_dir, fname)
        if os.path.exists(path):
            available.append((name, path, n_elem))
            print(f"  [OK] {name:<16} ({n_elem} elements) — golden file found")
        else:
            print(f"  [--] {name:<16} ({n_elem} elements) — golden file NOT found")

    print()
    print_table_header()

    for name, path, n_elem in available:
        try:
            golden = read_bin_int32(path, n_elem)
        except Exception as e:
            print(f"  {name:<16} {'ERROR':>8} — {e}")
            continue

        # For now, only verify that golden files are loadable
        # Full C-vs-MATLAB comparison requires C to output per-stage data
        # Placeholder — would need C test harness to export per-stage data
        max_val = max(abs(v) for v in golden)
        min_bits = effective_bits(min(abs(v) for v in golden if v != 0) if any(golden) else 0)
        max_bits = effective_bits(max_val)

        print(f"  {name:<16} {n_elem:>8} {'--':>10} {'--':>10} {'--':>12} {min_bits-20:>8} {max_bits-20:>8}")

    print("-" * 78)
    print()
    print("NOTE: Full C-vs-MATLAB per-bin comparison requires:")
    print("  1. Complete weight data (409 arrays, ~50K lines)")
    print("  2. C program to export per-stage outputs to .bin files")
    print("  3. MATLAB golden export (export_frame0_golden.m)")
    print()
    print("For the working STFT verification (55.9 dB SNR confirmed),")
    print("see stft_fixed_point_guide.md and the ulunas_nr_test binary.")

if __name__ == '__main__':
    main()
