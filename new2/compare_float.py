#!/usr/bin/env python3
"""Compare C vs MATLAB float dumps for all layers."""
import re, math, sys

def parse_arrays(text):
    arrays = {}
    for m in re.finditer(r'(\w+)\s*=\s*\[([^\]]+)\]', text):
        name = m.group(1)
        vals = [float(x.strip()) for x in m.group(2).split(',') if x.strip()]
        arrays[name] = vals
    return arrays

m_file = sys.argv[1] if len(sys.argv) > 1 else 'frame0_layer_io.txt'
c_file = sys.argv[2] if len(sys.argv) > 2 else 'c_float_dump.txt'

with open(m_file, 'r') as f: m_text = f.read()
with open(c_file, 'r') as f: c_text = f.read()

m_arrays = parse_arrays(m_text)
c_arrays = parse_arrays(c_text)

layers = ['STFT_real', 'STFT_imag', 'log_gen', 'BM', 'E0', 'E1', 'E2', 'E3', 'E4',
          'RNN1', 'RNN2', 'Decoder', 'Sigmoid', 'BS', 'MASK_real', 'MASK_imag']

# 2D tensor shapes: (name, C, W) for col-major -> row-major conversion
SHAPES_2D = {'E0': (12,65), 'E1': (24,33), 'E2': (24,33), 'E3': (32,33), 'E4': (16,33),
             'RNN1': (16,33), 'RNN2': (16,33)}

def col2row(data, C, W):
    result = [0.0] * (C * W)
    for c in range(C):
        for w in range(W):
            result[c * W + w] = data[c + C * w]
    return result

print(f"{'Layer':<12} {'Elements':>8} {'SNR(dB)':>10} {'MaxErr':>10} {'Status':>10}")
print("-" * 56)

for name in layers:
    if name not in m_arrays or name not in c_arrays:
        print(f"{name:<12} {'SKIP':>8}")
        continue
    m = m_arrays[name]; c = c_arrays[name]
    n = min(len(m), len(c)); m_raw, c_raw = m[:n], c[:n]

    # Convert MATLAB col-major to row-major for 2D tensors
    if name in SHAPES_2D:
        C, W = SHAPES_2D[name]
        m = col2row(m_raw, C, W)
        c = c_raw  # C is already row-major
    else:
        m, c = m_raw, c_raw

    signal = sum(x*x for x in m)
    noise = sum((m[i]-c[i])**2 for i in range(n))
    snr = 999.0 if noise < 1e-30 else 10.0 * math.log10(signal / noise)
    max_err = max(abs(m[i]-c[i]) for i in range(n))
    status = "PERFECT" if snr > 120 else ("PASS" if snr > 60 else ("WARN" if snr > 30 else "FAIL"))
    print(f"{name:<12} {n:>8} {snr:>10.2f} {max_err:>10.6f} {status:>10}")

print("\n=== Per-element samples (first 8) ===")
for name in ['log_gen', 'BM', 'E0', 'E1', 'Decoder']:
    if name in m_arrays and name in c_arrays:
        m = m_arrays[name]; c = c_arrays[name]
        print(f"\n{name}:")
        for i in range(min(8, len(m))):
            print(f"  [{i}] C={c[i]:.6f} M={m[i]:.6f} diff={abs(c[i]-m[i]):.6f}")
