#!/usr/bin/env python3
"""Frame 0 C vs MATLAB per-bin comparison for ALL layers (layout-corrected)."""
import sys, os, struct, math

def read_i32(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}i', f.read(n*4)))
def read_u16(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}H', f.read(n*2)))
def read_i16(path, n):
    with open(path, 'rb') as f:
        return list(struct.unpack(f'<{n}h', f.read(n*2)))

def snr_db(golden, test):
    s = sum(g*g for g in golden)
    e = sum((g-t)**2 for g,t in zip(golden, test))
    if e < 1e-30 or s < 1e-30: return 999.0
    return 10.0 * math.log10(s / e)

def col2row(data, C, W):
    result = [0] * (C * W)
    for c in range(C):
        for w in range(W):
            result[c * W + w] = data[c + C * w]
    return result

def print_table(golden, test, n_show=16, fmt='i32'):
    n = len(golden)
    indices = list(range(0, min(8, n))) + list(range(n//4, n//4+4)) + list(range(n-4, n))
    indices = sorted(set(i for i in indices if 0 <= i < n))[:n_show]
    print(f"{'Idx':>4} | {'C_Q20':>10} | {'G_Q20':>10} | {'Diff':>8} | {'Float_C':>12} | {'Float_G':>12}")
    print("-" * 74)
    for i in indices:
        c, g = test[i], golden[i]
        d = c - g
        fc = c / 1048576.0
        fg = g / 1048576.0
        print(f"{i:4} | {c:10} | {g:10} | {d:+8} | {fc:12.6f} | {fg:12.6f}")
    print("-" * 74)

def main():
    c_dir = sys.argv[1] if len(sys.argv) > 1 else "dump_c"
    g_dir = sys.argv[2] if len(sys.argv) > 2 else "dump_matlab"

    stages = [
        ("BM",       "frame0_bm_c.bin",       "frame0_bm.bin",       129,  False, 0, 0, 'i32'),
        ("E0",       "frame0_e0_xconv_c.bin", "frame0_enc_e0.bin",   780,  True,  12, 65, 'i32'),
        ("E1",       "frame0_e1_xmb0_c.bin",  "frame0_enc_e1.bin",   792,  True,  24, 33, 'i32'),
        ("E2",       "frame0_e2_xdws0_c.bin", "frame0_enc_e2.bin",   792,  True,  24, 33, 'i32'),
        ("E3",       "frame0_e3_xmb1_c.bin",  "frame0_enc_e3.bin",   1056, True,  32, 33, 'i32'),
        ("E4",       "frame0_e4_xdws1_c.bin", "frame0_enc_e4.bin",   528,  True,  16, 33, 'i32'),
        ("RNN1",     "frame0_gdprnn1_c.bin",  "frame0_rnn1.bin",     528,  True,  16, 33, 'i32'),
        ("RNN2",     "frame0_gdprnn2_c.bin",  "frame0_rnn2.bin",     528,  True,  16, 33, 'i32'),
        ("Decoder",  "frame0_decoder_c.bin",  "frame0_dec.bin",      129,  False, 0, 0, 'i32'),
        ("Sigmoid",  "frame0_sigmoid_c.bin",  "frame0_sig.bin",      129,  False, 0, 0, 'u16'),
        ("BS",       "frame0_bs_c.bin",       "frame0_bs.bin",       257,  False, 0, 0, 'i16'),
        ("log_gen",  "frame0_log_gen_c.bin",  "frame0_bm.bin",       129,  False, 0, 0, 'i32'),
    ]

    for name, c_file, g_file, n_elem, is_2d, C, W, dtype in stages:
        c_path = os.path.join(c_dir, c_file)
        g_path = os.path.join(g_dir, g_file)
        if not os.path.exists(c_path) or not os.path.exists(g_path):
            continue
        if dtype == 'i32':
            c_data, g_data = read_i32(c_path, n_elem), read_i32(g_path, n_elem)
        elif dtype == 'u16':
            c_data, g_data = read_u16(c_path, n_elem), read_u16(g_path, n_elem)
        elif dtype == 'i16':
            c_data, g_data = read_i16(c_path, n_elem), read_i16(g_path, n_elem)
        else:
            continue
        if is_2d:
            g_flat = col2row(g_data, C, W)
            c_flat = c_data
        else:
            g_flat = g_data
            c_flat = c_data
        snr = snr_db(g_flat, c_flat)
        dim_str = f"[{C}x{W}]" if is_2d else f"[{n_elem}]"
        print(f"\n### {name} {dim_str} — SNR={snr:.2f} dB")
        print_table(g_flat, c_flat)

if __name__ == '__main__':
    main()
