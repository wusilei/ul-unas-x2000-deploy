#!/usr/bin/env python3
"""Full per-element C vs MATLAB comparison for all layers."""
import re, math, sys

def parse_arrays(text):
    arrays = {}
    for m in re.finditer(r'(\w+)\s*=\s*\[([^\]]+)\]', text):
        name = m.group(1)
        vals = [float(x.strip()) for x in m.group(2).split(',') if x.strip()]
        arrays[name] = vals
    return arrays

# 2D tensor shapes for layout conversion
SHAPES_2D = {'E0': (12,65), 'E1': (24,33), 'E2': (24,33), 'E3': (32,33), 'E4': (16,33),
             'RNN1': (16,33), 'RNN2': (16,33)}

def col2row(data, C, W):
    result = [0.0] * (C * W)
    for c in range(C):
        for w in range(W):
            result[c * W + w] = data[c + C * w]
    return result

def snr(m, c):
    s = sum(x*x for x in m)
    e = sum((m[i]-c[i])**2 for i in range(len(m)))
    return 999.0 if e < 1e-30 else 10.0 * math.log10(s / e)

m_file = sys.argv[1] if len(sys.argv) > 1 else 'frame0_layer_io.txt'
c_file = sys.argv[2] if len(sys.argv) > 2 else 'c_float_dump.txt'
out_file = sys.argv[3] if len(sys.argv) > 3 else 'per_element_comparison.txt'

with open(m_file, 'r') as f: m_text = f.read()
with open(c_file, 'r') as f: c_text = f.read()

m_arrays = parse_arrays(m_text)
c_arrays = parse_arrays(c_text)

layers = [
    ('STFT_real', 'STFT_real'),
    ('STFT_imag', 'STFT_imag'),
    ('log_gen', 'log_gen'),
    ('BM', 'BM'),
    ('E0', 'E0'),
    ('E1', 'E1'),
    ('E2', 'E2'),
    ('E3', 'E3'),
    ('E4', 'E4'),
    ('RNN1', 'RNN1'),
    ('RNN2', 'RNN2'),
    ('Decoder', 'Decoder'),
    ('Sigmoid', 'Sigmoid'),
    ('BS', 'BS'),
    ('MASK_real', 'MASK_real'),
    ('MASK_imag', 'MASK_imag'),
]

with open(out_file, 'w') as f:
    f.write("=== UL-UNAS Frame 0: C vs MATLAB Per-Element Float Comparison ===\n")
    f.write("C: clean api13 source compiled with gcc 9.4.0\n")
    f.write("M: MATLAB R2020b, Fix_point + float pipeline\n\n")

    for m_name, c_name in layers:
        if m_name not in m_arrays or c_name not in c_arrays:
            f.write(f"\n=== [{m_name}] SKIP ===\n")
            continue

        m_raw = m_arrays[m_name]
        c_raw = c_arrays[c_name]
        n = min(len(m_raw), len(c_raw))
        m_raw, c_raw = m_raw[:n], c_raw[:n]

        # Layout conversion
        if m_name in SHAPES_2D:
            C, W = SHAPES_2D[m_name]
            m_vals = col2row(m_raw, C, W)
            c_vals = c_raw
            dim_str = f"[{C}x{W}]"
        else:
            m_vals, c_vals = m_raw, c_raw
            dim_str = f"[{n}]"

        s = snr(m_vals, c_vals)

        f.write(f"\n{'='*80}\n")
        f.write(f"[{m_name}] {dim_str} — SNR = {s:.2f} dB\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Idx':>6} | {'C_float':>14} | {'M_float':>14} | {'Diff':>14} | {'Diff%':>10}\n")
        f.write(f"{'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}\n")

        for i in range(n):
            cv, mv = c_vals[i], m_vals[i]
            diff = cv - mv
            pct = 100.0 * abs(diff) / (abs(mv) + 1e-30) if abs(mv) > 1e-30 else 0.0
            f.write(f"{i:6} | {cv:14.8f} | {mv:14.8f} | {diff:14.8f} | {pct:9.4f}%\n")

        max_i = max(range(n), key=lambda i: abs(c_vals[i] - m_vals[i]))
        f.write(f"\nMax error at idx {max_i}: C={c_vals[max_i]:.8f} M={m_vals[max_i]:.8f} diff={abs(c_vals[max_i]-m_vals[max_i]):.8f}\n")

    f.write(f"\n{'='*80}\n")
    f.write("End of comparison.\n")

print(f"Written: {out_file} ({sum(1 for _ in open(out_file))} lines)")
