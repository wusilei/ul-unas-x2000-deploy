#!/usr/bin/env python3
"""
calibrate_qr.py — PESQ/STOI-based decoder QR calibration
=========================================================
Lock encoder + DPRNN QR at MATLAB theory values.
Optimize decoder cTFA QR values (D0-D4) via coordinate descent.

Usage: python3 calibrate_qr.py <noisy.wav> <clean.wav> <work_dir> [--target pesq|stoi|both] [infer_bin]
"""
import sys, os, subprocess
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

N_FFT = 512; HOP = N_FFT // 2; WINDOW = np.hanning(N_FFT)
TARGET_SR = 16000

# Start from current best (PESQ-optimized)
BEST_QR = {
    'D0': {'ta': (-13, -8, -9),  'fa': (-13, -8, -9)},
    'D1': {'ta': (-11, -14, -9), 'fa': (-20, -14, -6)},
    'D2': {'ta': (-11, -14, -5), 'fa': (-8, -5, -2)},
    'D3': {'ta': (-4, -16, -4),  'fa': (-6, -4, -12)},
    'D4': {'ta': (-3, -15, -6),  'fa': (-12, -10, -8)},
}

def write_qr_config(qr_dict, work_dir):
    buf = []
    for name in ['D0', 'D1', 'D2', 'D3', 'D4']:
        buf.extend(qr_dict[name]['ta'])
        buf.extend(qr_dict[name]['fa'])
    np.array(buf, dtype=np.int32).tofile(os.path.join(work_dir, 'qr_config.bin'))

def precompute_stft(noisy_wav, work_dir):
    audio, sr = sf.read(noisy_wav)
    if audio.ndim > 1: audio = audio[:, 0]
    audio = audio.astype(np.float32)
    pad_len = N_FFT // 2
    front_pad = audio[1:pad_len+1][::-1] if len(audio) > pad_len+1 else np.zeros(pad_len, dtype=np.float32)
    back_pad = audio[-pad_len-1:-1][::-1] if len(audio) > pad_len+1 else np.zeros(pad_len, dtype=np.float32)
    audio_padded = np.concatenate([front_pad, audio, back_pad])
    n_frames = max(1, (len(audio_padded) - N_FFT) // HOP + 1)
    os.makedirs(work_dir, exist_ok=True)
    for f in range(n_frames):
        start = f * HOP
        spec = np.fft.rfft(audio_padded[start:start+N_FFT] * WINDOW)
        spec.real.astype(np.float32).tofile(f"{work_dir}/frame{f}_stft_real.bin")
        spec.imag.astype(np.float32).tofile(f"{work_dir}/frame{f}_stft_imag.bin")
    return sr, n_frames, pad_len

def run_eval(clean_wav, work_dir, infer_bin, sr, n_frames, pad_len, dec_bias=0):
    """Run C inference + ISTFT. Returns (pesq, stoi)."""
    cmd = [infer_bin, work_dir, work_dir, str(n_frames)]
    if dec_bias != 0:
        cmd.append(str(dec_bias))
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if ret.returncode != 0:
        return float('nan'), float('nan')

    # ISTFT
    enhanced = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float32)
    window_sum = np.zeros_like(enhanced)
    for f in range(n_frames):
        mask_path = f"{work_dir}/frame{f}_mask.bin"
        if not os.path.exists(mask_path): break
        data = np.fromfile(mask_path, dtype=np.int32)
        if len(data) != 514: continue
        spec_f = (data[:257] + 1j * data[257:]).astype(np.float64) / (2**20)
        frame_td = np.fft.irfft(spec_f, n=N_FFT) * WINDOW
        start = f * HOP
        enhanced[start:start+N_FFT] += frame_td
        window_sum[start:start+N_FFT] += WINDOW ** 2
    enhanced[window_sum > 1e-10] /= window_sum[window_sum > 1e-10]
    if len(enhanced) > 2 * pad_len:
        enhanced = enhanced[pad_len:-pad_len]

    # Align with clean
    clean_audio, _ = sf.read(clean_wav)
    if clean_audio.ndim > 1: clean_audio = clean_audio[:, 0]
    min_len = min(len(enhanced), len(clean_audio))
    enhanced, clean_audio = enhanced[:min_len], clean_audio[:min_len].astype(np.float64)

    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, TARGET_SR)
        enhanced = resample_poly(enhanced, TARGET_SR//g, sr//g)
        clean_audio = resample_poly(clean_audio, TARGET_SR//g, sr//g)

    try:
        p = pesq(TARGET_SR, clean_audio, enhanced, 'wb')
        s = stoi(clean_audio, enhanced, TARGET_SR, extended=False)
    except:
        p, s = float('nan'), float('nan')
    return p, s

def score_fn(pesq_val, stoi_val, target, base_stoi):
    """Combined score: higher is better."""
    if target == 'pesq':
        return pesq_val
    elif target == 'stoi':
        return stoi_val
    elif target == 'both':
        # PESQ normalized to ~[0,1] range (max ~4.5), equally weighted with STOI
        return pesq_val / 4.5 + stoi_val
    return pesq_val

def coordinate_descent(clean_wav, work_dir, infer_bin, qr_dict, sr, n_frames, pad_len,
                       target='pesq', rounds=2):
    best_qr = {k: {k2: v2 for k2, v2 in v.items()} for k, v in qr_dict.items()}
    best_bias = 0  # Q20 decoder bias

    qr_path = os.path.join(work_dir, 'qr_config.bin')
    if os.path.exists(qr_path): os.remove(qr_path)
    base_p, base_s = run_eval(clean_wav, work_dir, infer_bin, sr, n_frames, pad_len, best_bias)
    base_score = score_fn(base_p, base_s, target, base_s)
    print(f"  Baseline: PESQ={base_p:.4f} STOI={base_s:.4f} score={base_score:.4f} [{target}]")

    # ── Step 0: Decoder bias (fast global search) ──
    print(f"\n  [Bias search]")
    for bias_m in range(0, 9):  # 0 to 8M Q20 (0 to ~8 float)
        bias = bias_m * 1000000
        p, s = run_eval(clean_wav, work_dir, infer_bin, sr, n_frames, pad_len, bias)
        sc = score_fn(p, s, target, base_s)
        if not np.isnan(p) and sc > best_score:
            best_score, best_bias = sc, bias
            print(f"    bias={bias//1000000}M  P={p:.4f} S={s:.4f} score={sc:.4f}  <-- BEST")
        else:
            marker = '  <-- BEST' if not np.isnan(p) and sc > base_score else ''
            if not np.isnan(p):
                print(f"    bias={bias//1000000}M  P={p:.4f} S={s:.4f} score={sc:.4f}{marker}")
    print(f"  Best bias: {best_bias} ({best_bias/1e6:.1f}M Q20 = {best_bias/2**20:.1f} float)")

    params = []
    for layer in ['D1', 'D0', 'D2', 'D3', 'D4']:  # D1 first (biggest impact)
        for gate in ['ta', 'fa']:
            for pidx in range(3):
                params.append((layer, gate, pidx))

    # ── Step 1+: QR coordinate descent with best bias fixed ──
    best_score = base_score
    for rd in range(rounds):
        improved = False
        for layer, gate, pidx in params:
            current = best_qr[layer][gate][pidx]
            best_val, best_cand_score = current, best_score

            for delta in range(-4, 5):
                if delta == 0: continue
                candidate = current + delta
                if candidate > -1 or candidate < -24: continue

                test_qr = {k: {k2: v2 for k2, v2 in v.items()} for k, v in best_qr.items()}
                lst = list(test_qr[layer][gate])
                lst[pidx] = candidate
                test_qr[layer][gate] = tuple(lst)

                write_qr_config(test_qr, work_dir)
                p, s = run_eval(clean_wav, work_dir, infer_bin, sr, n_frames, pad_len, best_bias)
                sc = score_fn(p, s, target, base_s)

                if not np.isnan(p) and sc > best_cand_score:
                    best_cand_score, best_val = sc, candidate
                    improved = True
                    print(f"  [{rd+1}] {layer}.{gate}[{pidx}]: {current:3d}→{candidate:3d}"
                          f"  P={p:.4f} S={s:.4f} score={sc:.4f}")

            if best_val != current:
                lst = list(best_qr[layer][gate])
                lst[pidx] = best_val
                best_qr[layer][gate] = tuple(lst)
                best_score = best_cand_score

        if not improved:
            print(f"  Round {rd+1}: converged")
            break

    return best_qr, best_score

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <noisy.wav> <clean.wav> <work_dir> "
              f"[--target pesq|stoi|both] [infer_bin]")
        sys.exit(1)

    noisy_wav, clean_wav, work_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    target, infer_bin = 'pesq', './infer_audio_fixed'
    for a in sys.argv[4:]:
        if a.startswith('--target='):
            target = a.split('=')[1]
        elif not a.startswith('--'):
            infer_bin = a

    print(f"=== QR Calibration [{target}] ===")
    print(f"  Optimizing D0-D4 decoder cTFA QR")

    print(f"[1] Pre-computing STFT...")
    sr, n_frames, pad_len = precompute_stft(noisy_wav, work_dir)
    print(f"  {n_frames} frames, {sr} Hz\n")

    print(f"[2] Coordinate descent...")
    best_qr, best_score = coordinate_descent(clean_wav, work_dir, infer_bin,
                                              BEST_QR, sr, n_frames, pad_len, target)

    # Final eval
    write_qr_config(best_qr, work_dir)
    fp, fs = run_eval(clean_wav, work_dir, infer_bin, sr, n_frames, pad_len, best_bias)
    print(f"\n=== Best QR ({target.upper()}={best_score:.4f}) → PESQ={fp:.4f} STOI={fs:.4f} ===")
    for name in ['D0', 'D1', 'D2', 'D3', 'D4']:
        ta, fa = best_qr[name]['ta'], best_qr[name]['fa']
        print(f"  {name}_TA {ta[0]:3d}, {ta[1]:3d}, {ta[2]:3d}")
        print(f"  {name}_FA {fa[0]:3d}, {fa[1]:3d}, {fa[2]:3d}")
