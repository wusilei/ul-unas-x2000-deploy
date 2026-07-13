#!/usr/bin/env python3
"""End-to-end denoising using C inference via ctypes + MATLAB-compatible STFT/ISTFT."""
import sys, ctypes
import numpy as np
import soundfile as sf

# Load C shared library
lib = ctypes.CDLL('./libdenoise.so')

class ulunas_state_t(ctypes.Structure):
    _fields_ = [
        ("conv_cache_e0", ctypes.c_int32 * (2*129)),
        ("conv_cache_e1", ctypes.c_int32 * (24*65)),
        ("conv_cache_e2", ctypes.c_int32 * (24*33)),
        ("conv_cache_d0", ctypes.c_int32 * (24*33)),
        ("conv_cache_d1", ctypes.c_int32 * (12*33)),
        ("conv_cache_d2", ctypes.c_int32 * (12*2*65)),
        ("tfa_cache_e0", ctypes.c_int16 * 24),
        ("tfa_cache_e1", ctypes.c_int16 * 48),
        ("tfa_cache_e2", ctypes.c_int16 * 48),
        ("tfa_cache_e3", ctypes.c_int16 * 64),
        ("tfa_cache_e4", ctypes.c_int16 * 32),
        ("tfa_cache_d0", ctypes.c_int16 * 64),
        ("tfa_cache_d1", ctypes.c_int16 * 48),
        ("tfa_cache_d2", ctypes.c_int16 * 48),
        ("tfa_cache_d3", ctypes.c_int16 * 24),
        ("tfa_cache_d4", ctypes.c_int16 * 2),
        ("inter_cache_0", ctypes.c_int16 * (33*16)),
        ("inter_cache_1", ctypes.c_int16 * (33*16)),
    ]

lib.ulunas_state_init.argtypes = [ctypes.POINTER(ulunas_state_t)]
lib.ulunas_infer_frame.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ulunas_state_t), ctypes.POINTER(ctypes.c_int32),
]


def matlab_stft(audio, n_fft=512, win_len=512, win_inc=256, hann_win=None):
    """MATLAB-compatible STFT: mirror-pad, Hann window, raw FFT (no 1/N scaling)."""
    if hann_win is None:
        hann_win = np.hanning(win_len + 1)[:win_len]  # symmetric Hann

    pad_len = n_fft // 2  # 256
    # MATLAB: [flip(x(2:pad_len+1)), x, flip(x(end-pad_len:end-1))]
    # 1-indexed: x(2:257)→flip = x[256],x[255],...,x[1] (0-based)
    #            x(end-256:end-1)→flip = x[L-2],x[L-3],...,x[L-257] (0-based)
    left_pad = audio[pad_len:0:-1]      # audio[256] down to audio[1]
    right_pad = audio[-2:-(pad_len+2):-1]  # audio[L-2] down to audio[L-257]
    x_pad = np.concatenate([left_pad, audio, right_pad])

    total_len = len(x_pad)
    n_frames = (total_len - win_len) // win_inc + 1
    n_bins = n_fft // 2 + 1  # 257

    spec_real = np.zeros((n_frames, n_bins), dtype=np.float32)
    spec_imag = np.zeros((n_frames, n_bins), dtype=np.float32)

    for t in range(n_frames):
        start = t * win_inc
        frame = x_pad[start:start + win_len] * hann_win
        fx = np.fft.fft(frame, n=n_fft)
        spec_real[t, :] = fx[:n_bins].real
        spec_imag[t, :] = fx[:n_bins].imag

    return spec_real, spec_imag, n_frames, hann_win


def matlab_istft(spec_real, spec_imag, n_fft=512, win_len=512, win_inc=256, hann_win=None):
    """MATLAB-compatible ISTFT: conjugate symmetry, IFFT, WOLA, unpad."""
    if hann_win is None:
        hann_win = np.hanning(win_len + 1)[:win_len]

    n_frames, n_bins = spec_real.shape
    output_len = (n_frames - 1) * win_inc + win_len

    enh = np.zeros(output_len, dtype=np.float64)
    win_sum = np.zeros(output_len, dtype=np.float64)

    for t in range(n_frames):
        full_spec = np.zeros(n_fft, dtype=np.complex128)
        full_spec[:n_bins] = spec_real[t] + 1j * spec_imag[t]
        # Conjugate symmetry for bins n_bins..n_fft-1
        full_spec[n_bins:] = np.conj(full_spec[1:n_fft//2][::-1])

        x_time = np.fft.ifft(full_spec, n=n_fft).real
        x_win = x_time * hann_win

        start = t * win_inc
        enh[start:start + win_len] += x_win
        win_sum[start:start + win_len] += hann_win ** 2

    win_sum[win_sum < 1e-8] = 1.0
    enh /= win_sum

    pad_len = n_fft // 2
    return enh[pad_len:-pad_len].astype(np.float32) if pad_len > 0 else enh.astype(np.float32)


def denoise_wav(input_path, output_path):
    """Denoise a WAV file using C inference + MATLAB-compatible STFT/ISTFT."""
    audio, rate = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    # MATLAB-compatible STFT
    n_fft, win_len, win_inc = 512, 512, 256
    hann_win = np.hanning(win_len + 1)[:win_len]
    spec_real, spec_imag, n_frames, _ = matlab_stft(audio, n_fft, win_len, win_inc, hann_win)

    print(f"Processing {n_frames} frames at {rate} Hz...")

    state = ulunas_state_t()
    lib.ulunas_state_init(ctypes.byref(state))

    mask_real = np.zeros((n_frames, 257), dtype=np.float32)
    mask_imag = np.zeros((n_frames, 257), dtype=np.float32)

    for i in range(n_frames):
        real_in = spec_real[i].astype(np.float32)
        imag_in = spec_imag[i].astype(np.float32)

        y_mask = (ctypes.c_int32 * 514)()
        lib.ulunas_infer_frame(
            real_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            imag_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(state),
            y_mask
        )

        # C output: real[0..256], imag[0..256]
        for j in range(257):
            mask_real[i, j] = float(y_mask[j]) / 1048576.0
            mask_imag[i, j] = float(y_mask[257 + j]) / 1048576.0

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames} frames done")

    # MATLAB-compatible ISTFT
    denoised = matlab_istft(mask_real, mask_imag, n_fft, win_len, win_inc, hann_win)
    denoised = np.clip(denoised, -1.0, 1.0)

    sf.write(output_path, denoised, rate)
    print(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.wav> <output.wav>")
        sys.exit(1)
    denoise_wav(sys.argv[1], sys.argv[2])
