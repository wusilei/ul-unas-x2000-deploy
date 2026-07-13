# linux_api11 — FROZEN (production baseline)

**Frozen on**: 2026-07-10
**Supersedes**: linux_api (v5) freeze from 2026-07-08
**Model**: UL-UNAS v2 (14 bugs fixed, full fixed-point)

**Binary**: `ulunas_q15` (~1.06MB, MIPS32R2 static)
**Deployed as**: `/data/tianhai/ulunas_q15` on X2000 (192.168.42.159)

**Pipeline**:
- iccom rf11→wf12, 400×int16 @ 50ms, 8kHz
- 8kHz 直通 (no up/downsample) → Q15 FFT → UL-UNAS fixed-point → Q15 IFFT → WOLA
- WIN_INC=200, 3/2 混合窗 WOLA
- 纯后置 AGC (NR→LMS→三级门控 200/30)
- agc_gate_decision() API 公开, 前置增益外部化

**Key files**:
- ulunas_linux.c — iccom main loop (能量检测+门控+前置1.5×)
- ulunas_fp.c/h — UL-UNAS model operators (fixed-point)
- ulunas_infer.c — ulunas_infer_frame() wrapper
- ulunas_lut.c/h — sigmoid/tanh/log10 LUTs + sqrt
- ulunas_matlab_weights.c/h — 409 weight arrays (~964KB)
- ulunas_modules.c — Encoder/Decoder/GDPRNN assembly
- ulunas_nr.c — STFT/WOLA pipeline (WIN_INC=200, 8k 直通)
- agc.c/h — voice_AGC LMS + 三级门控 (200/30) + agc_gate_decision API
- fft_q15.h — Q15 512-pt FFT (forward + inverse)
- qr_config.h — per-layer Qr shift parameters
- layer_dims.h — network dimensions
- noise_reduction.h — interface header

**AGC 拓扑**: 纯后置 (NR→LMS→门控), 前置 1.5× 在 ulunas_linux.c 中
**三级门控**: 近场>200/远场>30/底噪 0.5×
**AGC API**: energy_calculate_and_smooth_s16() + agc_gate_decision() 公开

**Build**: `make`
**Deploy**: `make deploy` (→192.168.42.159:/data/tianhai/ulunas_q15)

**Signal control**:
- kill -USR1 <pid>  toggle denoise ON/OFF
- kill -USR2 <pid>  toggle AGC ON/OFF

**Performance** (MIPS32R2 @ soft-float):
- Binary: ~1.06MB static
- WIN_INC=200, 每 call 400 samples = 2 帧精确对齐
- RTF: ~1.27x

**已知局限**: WIN_INC=200 + 8k 直通 → DSP 时序敏感, 跨设备一致性差
**解决方案**: linux_api12 (8k→16k 上采样 + WIN_INC=256)

**DO NOT MODIFY** — frozen 2026-07-10. 后续开发在 linux_api12+.
