# linux_api10 — FROZEN from linux_api9

**Frozen on**: 2026-07-09
**Base**: linux_api9 (2026-07-08)
**Model**: UL-UNAS v2 (10 bugs fixed, full fixed-point)

**Changes from linux_api9**:
- (pending — user to specify)

**Qr alignment with MATLAB (verified 2026-07-09)**:
linux_api9's qr_config.h was already correct for Decoder FA GRU:
- D0/D1 FA GRU: -12, -7 ✅
- D3 FA GRU: -11, -6 ✅
- The Qr mismatch (-21,-16 vs -12,-7) found during golden analysis was in the v3 development branch, not here.

**Binary**: `ulunas_q15` (~1.1MB, MIPS32R2 static)
**Deployed as**: `/data/ulunas_q15` on X2000 (192.168.42.159)

**Pipeline**:
- iccom rf11→wf12, 400×int16 @ 50ms, 8kHz
- 8k→16k upsample → Q15 FFT → UL-UNAS float inf → Q15 IFFT → 16k→8k downsample
- Pre-AGC → UL-UNAS NR → output
- Float-free log_gen (int64 sqrt + LUT log10)
- Float-free ln_func (int64 stats + sqrt_q40_to_q20)

**Key files**:
- ulunas_linux.c — iccom main loop + AGC + STFT/ISTFT
- ulunas_fp.c/h — UL-UNAS model operators (fixed-point)
- ulunas_infer.c — ulunas_infer_frame() wrapper
- ulunas_lut.c/h — sigmoid/tanh/log10 LUTs + sqrt
- ulunas_matlab_weights.c/h — model weights (~944KB)
- ulunas_modules.c — Encoder/Decoder/GDPRNN assembly
- agc.c/h — voice_AGC LMS
- fft_q15.h — Q15 512-pt FFT (forward + inverse)
- qr_config.h — per-layer Qr shift parameters
- layer_dims.h — network dimensions
- noise_reduction.h — interface header

**Build**:
```
make
```

**Deploy**:
```
make deploy
```

**Signal control**:
- kill -USR1 <pid>  toggle denoise ON/OFF
- kill -USR2 <pid>  toggle AGC ON/OFF

**Performance** (MIPS32R2 @ soft-float):
- Memory: 800KB RSS
- Latency: ~32ms/frame (200smp @ 8kHz)
- RTF: ~1.27x (near real-time)

**Bugs fixed** (10 total):
1. state_init outside frame loop
2. gru_module sum-then-round
3. Golden col-major→row-major
4. pconv2d_func weight_stride
5. BN weight uint16 saturation
6. TA FC Qr (7 layers)
7. ln_func channel index
8. inter_rnn_module per-timestep states
9. Decoder FA GRU Qr
10. non_gtconv2d_func kernel reversal

**DO NOT MODIFY** — this is the production baseline.
