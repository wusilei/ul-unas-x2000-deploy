#!/usr/bin/env python3
"""
diag_full_audio.py — Full-audio C vs MATLAB frame-by-frame MASK SNR
====================================================================
Runs both C and MATLAB on the same noisy wav, dumps per-frame MASK,
and tracks SNR degradation over time.

Usage: python3 diag_full_audio.py <noisy.wav> <work_dir> [matlab_bin] [infer_bin]
"""
import sys, os, subprocess, struct
import numpy as np
import soundfile as sf

N_FFT = 512; HOP = N_FFT // 2; WINDOW = np.hanning(N_FFT)

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
    return sr, n_frames

def run_c_model(infer_bin, work_dir, n_frames):
    """Run C infer_audio, return frame-level MASK SNR (self-check)."""
    # Clean old MASK files
    for f in range(n_frames):
        p = f"{work_dir}/frame{f}_mask.bin"
        if os.path.exists(p): os.remove(p)
        p = f"{work_dir}/frame{f}_dec.bin"
        if os.path.exists(p): os.remove(p)

    ret = subprocess.run([infer_bin, work_dir, work_dir, str(n_frames)],
                         capture_output=True, text=True, timeout=600)
    if ret.returncode != 0:
        print(f"  C model ERROR: {ret.stderr}")
        return None, None

    # Load all C MASK outputs
    c_masks = []
    c_decs = []
    for f in range(n_frames):
        mask_path = f"{work_dir}/frame{f}_mask.bin"
        dec_path = f"{work_dir}/frame{f}_dec.bin"
        if not os.path.exists(mask_path): break
        c_masks.append(np.fromfile(mask_path, dtype=np.int32))
        if os.path.exists(dec_path):
            c_decs.append(np.fromfile(dec_path, dtype=np.int32))
    return c_masks, c_decs

def run_matlab_model(work_dir, n_frames):
    """Generate MATLAB script and run it to produce MASK golden."""
    matlab_script = f"""
    addpath('{os.getcwd()}');
    addpath('para_in_mat_FP');
    addpath('test_wavs');

    % Load ERB
    erbfc_weight = importdata('erb_erb_fc_weight.mat');
    ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

    % Init caches (match Main_infer.m)
    conv_cache_e0 = zeros(2,129);
    conv_cache_e1 = zeros(24,65);
    conv_cache_e2 = zeros(24,33);
    conv_cache_d0 = zeros(24,33);
    conv_cache_d1 = zeros(12,33);
    conv_cache_d2 = zeros(12,2,65);
    tfa_cache_e0 = zeros(1,24);
    tfa_cache_e1 = zeros(1,48);
    tfa_cache_e2 = zeros(1,48);
    tfa_cache_e3 = zeros(1,64);
    tfa_cache_e4 = zeros(1,32);
    tfa_cache_d0 = zeros(1,64);
    tfa_cache_d1 = zeros(1,48);
    tfa_cache_d2 = zeros(1,48);
    tfa_cache_d3 = zeros(1,24);
    tfa_cache_d4 = zeros(1,2);
    inter_cache_0 = zeros(33,16);
    inter_cache_1 = zeros(33,16);

    work_dir = '{work_dir}';
    n_frames = {n_frames};

    for f = 0:n_frames-1
        % Load STFT frame
        fid = fopen(sprintf('%s/frame%d_stft_real.bin', work_dir, f), 'rb');
        real_in = fread(fid, 257, 'float'); fclose(fid);
        fid = fopen(sprintf('%s/frame%d_stft_imag.bin', work_dir, f), 'rb');
        imag_in = fread(fid, 257, 'float'); fclose(fid);

        % Log-gen
        x_log = log_gen(real_in, imag_in);
        x_log = Fix_point(x_log, 's32f20');

        % BM
        x_bm = BM_module(x_log, erbfc_weight);

        % Encoder
        [y_e0, conv_cache_e0, tfa_cache_e0] = XConv_module(x_bm, conv_cache_e0, tfa_cache_e0);
        [y_e1, tfa_cache_e1] = XMB0_module(y_e0, tfa_cache_e1);
        [y_e2, conv_cache_e1, tfa_cache_e2] = XDWS0_module(y_e1, conv_cache_e1, tfa_cache_e2);
        [y_e3, tfa_cache_e3] = XMB1_module(y_e2, tfa_cache_e3);
        [y_e4, conv_cache_e2, tfa_cache_e4] = XDWS1_module(y_e3, conv_cache_e2, tfa_cache_e4);

        % GDPRNN
        y_intra = Intra_RNN_module(y_e4.', 0);
        [y_inter, inter_cache_0] = Inter_RNN_module(y_intra, inter_cache_0, 0);
        y_rnn1 = y_inter.';
        y_intra = Intra_RNN_module(y_rnn1.', 1);
        [y_inter, inter_cache_1] = Inter_RNN_module(y_intra, inter_cache_1, 1);
        y_rnn2 = y_inter.';

        % Decoder
        [y_d0, tfa_cache_d0] = De_XDWS0_module(y_rnn2, y_e4, tfa_cache_d0);
        [y_d1, tfa_cache_d1] = De_XMB0_module(y_d0, y_e3, tfa_cache_d1);
        [y_d2, conv_cache_d0, tfa_cache_d2] = De_XDWS1_module(y_d1, y_e2, conv_cache_d0, tfa_cache_d2);
        [y_d3, conv_cache_d1, tfa_cache_d3] = De_XMB1_module(y_d2, y_e1, conv_cache_d1, tfa_cache_d3);
        [y_dec, conv_cache_d2, tfa_cache_d4] = De_XConv_module(y_d3, y_e0, conv_cache_d2, tfa_cache_d4);

        % Save decoder output
        fid = fopen(sprintf('%s/frame%d_mat_dec.bin', work_dir, f), 'wb');
        fwrite(fid, y_dec, 'int32'); fclose(fid);

        % Sigmoid → BS → MASK
        y_dec_dq = y_dec * 2^(-20);
        y_sig = sigmoid_func(y_dec_dq);
        y_sig = Fix_point(y_sig, 'u16f15');
        y_bs = BS_module(y_sig, ierbfc_weight);
        y_bs_fp = double(y_bs) / 32768;

        real_q = Fix_point(real_in, 's32f20');
        imag_q = Fix_point(imag_in, 's32f20');
        y_mask = MASK_module(y_bs_fp, real_q, imag_q);
        % MASK_module returns float after *2^(-20), need Q20 int32 for comparison
        y_mask_q = Fix_point(y_mask, 's32f20');

        % Save MASK output
        fid = fopen(sprintf('%s/frame%d_mat_mask.bin', work_dir, f), 'wb');
        fwrite(fid, y_mask_q, 'int32'); fclose(fid);

        if mod(f, 50) == 0
            fprintf('  MATLAB frame %d/%d\\n', f, n_frames);
        end
    end
    fprintf('MATLAB done: %d frames\\n', n_frames);
    quit;
    """

    # Write MATLAB script
    script_path = f"{work_dir}/diag_full_mask.m"
    with open(script_path, 'w') as f:
        f.write(matlab_script)

    # Run MATLAB
    print(f"  Running MATLAB ({n_frames} frames)...")
    ret = subprocess.run(
        ['/e/app/matlab/bin/matlab', '-nodisplay', '-nosplash', '-nodesktop',
         '-batch', f"run('{script_path}')"],
        capture_output=True, text=True, timeout=3600,
        cwd=work_dir
    )
    if ret.returncode != 0:
        # Try to extract error info
        err_lines = [l for l in ret.stdout.split('\n') + ret.stderr.split('\n') if 'Error' in l or 'error' in l]
        print(f"  MATLAB error: {'; '.join(err_lines[:3])}")
        return None, None

    # Load MATLAB MASK outputs
    mat_masks = []
    mat_decs = []
    for f in range(n_frames):
        mask_path = f"{work_dir}/frame{f}_mat_mask.bin"
        if not os.path.exists(mask_path): break
        mat_masks.append(np.fromfile(mask_path, dtype=np.int32))
        dec_path = f"{work_dir}/frame{f}_mat_dec.bin"
        if os.path.exists(dec_path):
            mat_decs.append(np.fromfile(dec_path, dtype=np.int32))
    return mat_masks, mat_decs

def snr_db(golden, test):
    s = np.sum(golden.astype(np.float64)**2)
    e = np.sum((golden.astype(np.float64) - test.astype(np.float64))**2)
    return 999.0 if e < 1e-30 else 10 * np.log10(s / e)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <noisy.wav> <work_dir> [infer_bin]")
        sys.exit(1)

    noisy_wav, work_dir = sys.argv[1], sys.argv[2]
    infer_bin = sys.argv[3] if len(sys.argv) > 3 else "./infer_audio_fixed"

    print(f"=== Full-Audio C vs MATLAB MASK Comparison ===\n")

    # Step 1: STFT
    print(f"[1/4] STFT analysis...")
    sr, n_frames = precompute_stft(noisy_wav, work_dir)
    print(f"  {n_frames} frames, {sr} Hz")

    # Step 2: C model
    print(f"\n[2/4] C model inference...")
    c_masks, c_decs = run_c_model(infer_bin, work_dir, n_frames)
    if c_masks is None:
        print("  FAILED")
        sys.exit(1)
    print(f"  {len(c_masks)} MASK frames")

    # Step 3: MATLAB model
    print(f"\n[3/4] MATLAB model inference...")
    mat_masks, mat_decs = run_matlab_model(work_dir, n_frames)
    if mat_masks is None or len(mat_masks) < 10:
        print("  FAILED — ensure MATLAB is available at /e/app/matlab/bin/matlab")
        sys.exit(1)
    print(f"  {len(mat_masks)} MASK frames")

    # Step 4: Compare
    print(f"\n[4/4] Frame-by-frame MASK SNR comparison...")
    n_compare = min(len(c_masks), len(mat_masks))
    mask_snrs = []
    dec_snrs = []

    for f in range(n_compare):
        ms = snr_db(mat_masks[f], c_masks[f])
        mask_snrs.append(ms)
        if f < len(c_decs) and f < len(mat_decs):
            ds = snr_db(mat_decs[f], c_decs[f])
            dec_snrs.append(ds)

    mask_snrs = np.array(mask_snrs)
    dec_snrs = np.array(dec_snrs) if dec_snrs else np.array([])

    # Summary
    print(f"\n  MASK SNR over {n_compare} frames:")
    print(f"    Mean:  {np.mean(mask_snrs):.2f} dB")
    print(f"    Std:   {np.std(mask_snrs):.2f} dB")
    print(f"    Min:   {np.min(mask_snrs):.2f} dB (frame {np.argmin(mask_snrs)})")
    print(f"    Max:   {np.max(mask_snrs):.2f} dB (frame {np.argmax(mask_snrs)})")

    # Check for drift: compare first 10% vs last 10%
    n10 = max(1, n_compare // 10)
    first_mean = np.mean(mask_snrs[:n10])
    last_mean = np.mean(mask_snrs[-n10:])
    drift = last_mean - first_mean
    print(f"    First {n10} frames: {first_mean:.2f} dB")
    print(f"    Last  {n10} frames: {last_mean:.2f} dB")
    print(f"    Drift: {drift:+.2f} dB  {'⚠️ STATE DRIFT' if drift < -2 else '✅ stable'}")

    # Segment analysis: every 50 frames
    print(f"\n  Segmented (every 50 frames):")
    for seg_start in range(0, n_compare, 50):
        seg_end = min(seg_start + 50, n_compare)
        seg_snr = np.mean(mask_snrs[seg_start:seg_end])
        bar = '█' * max(0, int(seg_snr + 20))
        print(f"    [{seg_start:4d}-{seg_end:4d}]: {seg_snr:6.2f} dB {bar}")

    if len(dec_snrs) > 0:
        print(f"\n  Decoder SNR over {len(dec_snrs)} frames:")
        print(f"    Mean: {np.mean(dec_snrs):.2f} dB  Min: {np.min(dec_snrs):.2f} dB  Max: {np.max(dec_snrs):.2f} dB")

    # Identify worst frames
    worst_idx = np.argsort(mask_snrs)[:5]
    print(f"\n  Worst 5 frames:")
    for idx in worst_idx:
        print(f"    Frame {idx}: MASK={mask_snrs[idx]:.2f} dB", end='')
        if idx < len(dec_snrs):
            print(f"  Dec={dec_snrs[idx]:.2f} dB")
        else:
            print()

    print(f"\n  Conclusion: ", end='')
    if drift < -3:
        print("SIGNIFICANT STATE DRIFT — DPRNN inter_cache degrades over time ⚠️")
    elif np.mean(mask_snrs) < -2:
        print("Systematic MASK offset — C vs MATLAB consistently differ ⚠️")
    elif np.mean(mask_snrs) > -1:
        print("C ≈ MATLAB — MASK SNR stable across full audio ✅")
    else:
        print("Moderate differences — check worst frames for patterns")

if __name__ == "__main__":
    main()
