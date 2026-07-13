#!/usr/bin/env python3
"""Read MATLAB golden files and print per-layer tables with known SNR."""
import struct, math, os, sys

def read_i32(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}i', f.read(n*4)))
def read_u16(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}H', f.read(n*2)))
def read_i16(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}h', f.read(n*2)))

def col2row(data, C, W):
    result = [0] * (C * W)
    for c in range(C):
        for w in range(W):
            result[c * W + w] = data[c + C * w]
    return result

g_dir = sys.argv[1] if len(sys.argv) > 1 else "dump_matlab"

# Layers: (name, file, n_elem, shape, SNR_dB, dtype)
LAYERS = [
    ("BM",       "frame0_bm.bin",        129,  None,    84.2, 'i32'),
    ("E0",       "frame0_enc_e0.bin",    780,  (12,65), 67.5, 'i32'),
    ("E1",       "frame0_enc_e1.bin",    792,  (24,33), 64.7, 'i32'),
    ("E2",       "frame0_enc_e2.bin",    792,  (24,33), 65.3, 'i32'),
    ("E3",       "frame0_enc_e3.bin",    1056, (32,33), 64.5, 'i32'),
    ("E4",       "frame0_enc_e4.bin",    528,  (16,33), 64.1, 'i32'),
    ("RNN1",     "frame0_rnn1.bin",      528,  (16,33), 67.5, 'i32'),
    ("RNN2",     "frame0_rnn2.bin",      528,  (16,33), 64.6, 'i32'),
    ("Decoder",  "frame0_dec.bin",       129,  None,    62.0, 'i32'),
    ("Sigmoid",  "frame0_sig.bin",       129,  None,    999,  'u16'),
    ("BS",       "frame0_bs.bin",        257,  None,    84.1, 'i16'),
]

for name, fname, n, shape, snr, dtype in LAYERS:
    path = os.path.join(g_dir, fname)
    if not os.path.exists(path):
        print(f"\n### [{name}] — SKIP (file not found: {fname})\n")
        continue

    if dtype == 'i32':
        data = read_i32(path, n)
    elif dtype == 'u16':
        data = read_u16(path, n)
    elif dtype == 'i16':
        data = read_i16(path, n)
    else:
        continue

    if shape:
        data = col2row(data, shape[0], shape[1])
        dim_str = f"[{shape[0]}x{shape[1]}]"
    else:
        dim_str = f"[{n}]"

    # Estimate typical diff from SNR
    if snr >= 999:
        typical_diff = 0
    else:
        rms = math.sqrt(sum(x*x for x in data) / len(data))
        rms_n = rms / (10 ** (snr / 20))
        typical_diff = int(rms_n)

    print(f"\n### [{name}] {dim_str} — SNR={snr} dB (typical C-M Diff ~{typical_diff} LSB)\n")

    N = len(data)
    idx = list(range(0, min(8, N))) + list(range(N//4, N//4+4)) + list(range(N-4, N))
    idx = sorted(set(i for i in idx if 0 <= i < N))[:16]

    if dtype == 'u16':
        print(f"{'Idx':>4} | {'G_u16':>8}")
        print("-" * 20)
        for i in idx:
            print(f"{i:4} | {data[i]:8}")
    elif dtype == 'i16':
        print(f"{'Idx':>4} | {'G_s16':>8} | {'Float':>12}")
        print("-" * 32)
        for i in idx:
            print(f"{i:4} | {data[i]:8} | {data[i]/32768.0:12.6f}")
    else:
        print(f"{'Idx':>4} | {'G_Q20':>12} | {'Float':>12}")
        print("-" * 36)
        for i in idx:
            print(f"{i:4} | {data[i]:12} | {data[i]/1048576.0:12.6f}")
    print()
