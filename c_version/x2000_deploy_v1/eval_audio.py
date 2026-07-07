#!/usr/bin/env python3
"""
eval_audio.py — End-to-end UL-UNAS audio evaluation with PESQ/STOI
===================================================================
Pipeline:
  1. Read noisy + clean wav
  2. STFT (hanning, 50% overlap, 512-pt FFT)
  3. Export STFT frames → C model → read MASK output
  4. ISTFT → save enhanced wav
  5. PESQ / STOI vs clean reference

Usage: python3 eval_audio.py <noisy.wav> <clean.wav> <work_dir>
"""
import sys, os, struct, subprocess
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

N_FFT = 512
N_BINS = N_FFT // 2 + 1  # 257
HOP = N_FFT // 2           # 256 (50% overlap)
WINDOW = np.hanning(N_FFT)

def wav_to_stft(wav_path, work_dir):
    """Read wav, compute STFT, save frames to binary.
    Mirror-padding matches MATLAB STFT_func.m: flip(x(2:257)) x flip(x(end-255:end-1))."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    audio = audio.astype(np.float32)

    # Mirror padding: matches MATLAB's STFT_func pad_len=N_fft/2=256
    # Front: flip(x[1:257])  Back: flip(x[end-256:end-1])
    pad_len = N_FFT // 2  # 256
    if len(audio) > pad_len + 1:
        front_pad = audio[1:pad_len+1][::-1]  # flip(x(2:pad_len+1))
        back_pad = audio[-pad_len-1:-1][::-1]  # flip(x(end-pad_len:end-1))
    else:
        front_pad = np.zeros(pad_len, dtype=np.float32)
        back_pad = np.zeros(pad_len, dtype=np.float32)
    audio_padded = np.concatenate([front_pad, audio, back_pad])

    # Number of frames from padded signal
    total_len = len(audio_padded)
    n_frames = (total_len - N_FFT) // HOP + 1
    if n_frames <= 0:
        n_frames = 1

    os.makedirs(work_dir, exist_ok=True)
    for f in range(n_frames):
        start = f * HOP
        frame = audio_padded[start:start + N_FFT] * WINDOW
        spec = np.fft.rfft(frame)  # complex, 257 bins
        real_part = spec.real.astype(np.float32)
        imag_part = spec.imag.astype(np.float32)
        real_part.tofile(f"{work_dir}/frame{f}_stft_real.bin")
        imag_part.tofile(f"{work_dir}/frame{f}_stft_imag.bin")

    return audio_padded, sr, n_frames

def mask_to_wav(work_dir, n_frames, out_wav, sr):
    """Read MASK output (complex Q20 int32), convert to float, ISTFT."""
    # Read MASK outputs and convert from Q20 to float complex
    enhanced = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float32)
    window_sum = np.zeros_like(enhanced)

    for f in range(n_frames):
        mask_path = f"{work_dir}/frame{f}_mask.bin"
        if not os.path.exists(mask_path):
            print(f"  Missing frame {f}, stopping ISTFT")
            break

        # Read CRM output: (2, 257) int32 Q20
        data = np.fromfile(mask_path, dtype=np.int32)
        if len(data) != 514:
            print(f"  Frame {f}: expected 514 int32, got {len(data)}")
            continue

        # C/MATLAB MASK format: [real_0..real_256, imag_0..imag_256] (NOT interleaved)
        real_q20 = data[:257].astype(np.float64)   # first 257 = real
        imag_q20 = data[257:].astype(np.float64)   # last 257 = imag
        spec_float = (real_q20 + 1j * imag_q20) / (2**20)  # Q20 → float

        # ISTFT for this frame
        frame_td = np.fft.irfft(spec_float, n=N_FFT) * WINDOW

        start = f * HOP
        enhanced[start:start + N_FFT] += frame_td
        window_sum[start:start + N_FFT] += WINDOW ** 2

    # Normalize by window overlap
    mask = window_sum > 1e-10
    enhanced[mask] /= window_sum[mask]

    # Remove zero-padding from first/last N_FFT/2 samples (matching MATLAB)
    pad_len = N_FFT // 2
    if len(enhanced) > 2 * pad_len:
        enhanced = enhanced[pad_len:-pad_len]

    sf.write(out_wav, enhanced.astype(np.float32), sr)
    return enhanced

def compute_metrics(enhanced, clean_wav, sr):
    """Compute PESQ (wb) and STOI."""
    clean_audio, _ = sf.read(clean_wav)
    if clean_audio.ndim > 1:
        clean_audio = clean_audio[:, 0]
    clean_audio = clean_audio.astype(np.float64)

    # Align lengths
    min_len = min(len(enhanced), len(clean_audio))
    enhanced = enhanced[:min_len]
    clean_audio = clean_audio[:min_len]

    # Resample to 16kHz if needed (PESQ requires 8k or 16k)
    target_sr = 16000
    if sr != target_sr:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(sr, target_sr)
        enhanced_16k = resample_poly(enhanced, target_sr // gcd, sr // gcd)
        clean_16k = resample_poly(clean_audio, target_sr // gcd, sr // gcd)
    else:
        enhanced_16k = enhanced
        clean_16k = clean_audio

    # PESQ wideband
    try:
        p = pesq(target_sr, clean_16k, enhanced_16k, 'wb')
    except Exception as e:
        print(f"  PESQ error: {e}")
        p = float('nan')

    # STOI
    try:
        s = stoi(clean_16k, enhanced_16k, target_sr, extended=False)
    except Exception as e:
        print(f"  STOI error: {e}")
        s = float('nan')

    return p, s

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <noisy.wav> <clean.wav> <work_dir> [infer_bin]")
        sys.exit(1)

    noisy_wav = sys.argv[1]
    clean_wav = sys.argv[2]
    work_dir = sys.argv[3]
    infer_bin = sys.argv[4] if len(sys.argv) > 4 else "./infer_audio"

    print(f"=== UL-UNAS Audio Eval ===")
    print(f"  Noisy: {noisy_wav}")
    print(f"  Clean: {clean_wav}")

    # Step 1: STFT
    print(f"\n[1/5] STFT analysis...")
    audio, sr, n_frames = wav_to_stft(noisy_wav, work_dir)
    print(f"  {n_frames} frames, {sr} Hz, {len(audio)/sr:.1f}s")

    # Step 2: C model inference
    print(f"\n[2/5] C model inference ({n_frames} frames)...")
    stft_dir = work_dir  # STFT already there
    mask_dir = work_dir   # MASK output goes here too
    ret = subprocess.run([infer_bin, stft_dir, mask_dir, str(n_frames)],
                         capture_output=True, text=True, timeout=300)
    if ret.returncode != 0:
        print(f"  ERROR: {ret.stderr}")
        sys.exit(1)
    print(f"  {ret.stdout.strip().split(chr(10))[-1]}")

    # Step 3: ISTFT
    print(f"\n[3/5] ISTFT synthesis...")
    out_wav = f"{work_dir}/enhanced.wav"
    enhanced = mask_to_wav(work_dir, n_frames, out_wav, sr)
    print(f"  Saved: {out_wav}")

    # Step 4: Metrics
    print(f"\n[4/5] PESQ/STOI...")
    pesq_score, stoi_score = compute_metrics(enhanced, clean_wav, sr)
    print(f"  PESQ (wb): {pesq_score:.3f}")
    print(f"  STOI:      {stoi_score:.3f}")

    # Step 5: Noisy baseline
    print(f"\n[5/5] Noisy baseline metrics...")
    noisy_audio, _ = sf.read(noisy_wav)
    if noisy_audio.ndim > 1: noisy_audio = noisy_audio[:, 0]
    min_len = min(len(noisy_audio), len(enhanced))
    p_n, s_n = compute_metrics(noisy_audio[:min_len], clean_wav, sr)
    print(f"  Noisy PESQ: {p_n:.3f}")
    print(f"  Noisy STOI: {s_n:.3f}")

    print(f"\n=== Results ===")
    print(f"  PESQ: {p_n:.3f} → {pesq_score:.3f} (Δ={pesq_score-p_n:+.3f})")
    print(f"  STOI: {s_n:.3f} → {stoi_score:.3f} (Δ={stoi_score-s_n:+.3f})")
