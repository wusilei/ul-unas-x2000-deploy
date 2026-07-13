#!/usr/bin/env python3
"""Generate per-bin C vs M comparison tables for all pipeline layers."""
import struct, math, os, sys

def ri(p,n):
    with open(p,'rb') as f: return list(struct.unpack('<%di'%n, f.read(n*4)))
def ru(p,n):
    with open(p,'rb') as f: return list(struct.unpack('<%dH'%n, f.read(n*2)))
def rs(p,n):
    with open(p,'rb') as f: return list(struct.unpack('<%dh'%n, f.read(n*2)))

def snr(g,c):
    s=sum(x*x for x in g); e=sum((g[i]-c[i])**2 for i in range(len(g)))
    return 999 if e<1e-30 else 10*math.log10(s/e)

def col2row(data, C, W):
    result = [0]*(C*W)
    for c in range(C):
        for w in range(W): result[c*W+w] = data[c+C*w]
    return result

GD = sys.argv[1] if len(sys.argv) > 1 else 'dump_matlab'
CD = sys.argv[2] if len(sys.argv) > 2 else 'dump_c'

layers = [
    ("BM",      "frame0_bm_c.bin",      "frame0_bm.bin",      129,  None,    'i32', 's32f20'),
    ("E0",      "frame0_e0_c.bin",      "frame0_enc_e0.bin",  780,  (12,65), 'i32', 's32f20'),
    ("E1",      "frame0_e1_c.bin",      "frame0_enc_e1.bin",  792,  (24,33), 'i32', 's32f20'),
    ("E2",      "frame0_e2_c.bin",      "frame0_enc_e2.bin",  792,  (24,33), 'i32', 's32f20'),
    ("E3",      "frame0_e3_c.bin",      "frame0_enc_e3.bin",  1056, (32,33), 'i32', 's32f20'),
    ("E4",      "frame0_e4_c.bin",      "frame0_enc_e4.bin",  528,  (16,33), 'i32', 's32f20'),
    ("RNN1",    "frame0_rnn1_c.bin",    "frame0_rnn1.bin",    528,  (16,33), 'i32', 's32f20'),
    ("RNN2",    "frame0_rnn2_c.bin",    "frame0_rnn2.bin",    528,  (16,33), 'i32', 's32f20'),
    ("Decoder", "frame0_dec_c.bin",     "frame0_dec.bin",     129,  None,    'i32', 's32f20'),
    ("Sigmoid", "frame0_sig_c.bin",     "frame0_sig.bin",     129,  None,    'u16', 'u16f15'),
    ("BS",      "frame0_bs_c.bin",      "frame0_bs.bin",      257,  None,    'i16', 's16f15'),
]

for name, cf, gf, n, shape, dtype, qfmt in layers:
    cp = os.path.join(CD, cf); gp = os.path.join(GD, gf)
    if not os.path.exists(cp) or not os.path.exists(gp):
        print(f"\n### [{name}] -- SKIP (file missing: {cf} or {gf})\n")
        continue

    if dtype == 'i32':
        c = ri(cp, n); g = ri(gp, n)
    elif dtype == 'u16':
        c = ru(cp, n); g = ru(gp, n)
    elif dtype == 'i16':
        c = rs(cp, n); g = rs(gp, n)
    else:
        continue

    if shape:
        g = col2row(g, shape[0], shape[1])
        dim = f"[{shape[0]}x{shape[1]}]"
    else:
        dim = f"[{n}]"

    s = snr(g, c)
    print(f"\n### [{name}] {dim} -- SNR={s:.2f} dB\n")

    print(f"| Idx | C {qfmt} | M {qfmt} | Diff LSB | Float_M |")
    print(f"|-----|---------|---------|---------|---------|")

    N = len(g)
    idx = list(range(0, min(8, N)))
    idx += list(range(N//4, N//4+4))
    idx += list(range(N-4, N)) if N > 16 else []
    idx = sorted(set(i for i in idx if 0 <= i < N))[:16]

    for i in idx:
        cv, gv = c[i], g[i]
        d = cv - gv
        if dtype in ('u16', 'i16'):
            fg = gv / 32768.0
            print(f"| {i:4} | {cv:7} | {gv:7} | {d:+7} | {fg:12.6f} |")
        else:
            fg = gv / 1048576.0
            print(f"| {i:4} | {cv:10} | {gv:10} | {d:+8} | {fg:12.6f} |")
    print()
