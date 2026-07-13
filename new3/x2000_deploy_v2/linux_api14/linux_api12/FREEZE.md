# linux_api12 — Current Development (8k→16k Upsample)

**Created**: 2026-07-10
**Based on**: linux_api11 (frozen production baseline)
**Status**: Development — 待 X2000 部署验证

## Change Summary

**移植 GTCRN 上下采样管线, 解决 WIN_INC=200 + 8k 直通的 DSP 时序敏感问题.**

| 项目 | linux_api11 (frozen) | linux_api12 (current) |
|------|---------------------|----------------------|
| WIN_INC | 200 | **256** |
| 采样率 | 8k 直通 | **8k→16k→8k** |
| 上采样 | 无 | 线性插值 2× |
| 降采样 | 无 | 均值 2:1 |
| WOLA | 3/2 混合窗 (200 项倒数表) | **标准 2 窗 (power-complementary, 无需归一化)** |
| FIFO 缓冲 | 无余量 (2 帧精确对齐) | **256 点缓冲余量** |
| DSP 时序抖动容限 | 低 | **高** |
| 跨设备一致性 | ❌ 差 | ✅ 好 (GTCRN 验证) |

## Pipeline

```
iccom rf11 → 400×int16 @ 8kHz (50ms)
    ↓
[8k→16k 上采样] 线性插值: 400→800 samples
    ↓
[FIFO] → STFT (WIN_INC=256, Hann窗)
    ↓
[UL-UNAS 推理] Q15 FFT → fixed-point NR → Q15 IFFT
    ↓
[OLA] 标准 2 窗重叠, 无需 WOLA 归一化
    ↓
[16k→8k 降采样] 均值: 800→400 samples
    ↓
[AGC] 纯后置 LMS + 三级门控 (200/30)
    ↓
iccom wf12 → 400×int16 @ 8kHz
```

## Key Changes in ulunas_nr.c

1. WIN_INC: 200 → 256
2. 新增 `g_last_in_8k` 状态变量
3. 新增 8k→16k 上采样循环 (线性插值, ~0.1ms)
4. 移除 200 项 WOLA 倒数表 (WIN_INC=256 下 Hann 窗 power-complementary, 无需归一化)
5. OLA 缓冲: 712 → 768 (WIN_LEN + WIN_INC)
6. 新增 16k→8k 降采样循环 (均值, ~0.1ms)
7. Warmup: 5+3 → 8+5 (更多帧等待 16kHz FIFO 填充)
8. 总开销增加 ~0.2ms/frame, 可忽略

## Other Files (unchanged from linux_api11)

- ulunas_linux.c, agc.c/h — 接口不变 (noise_reduction(in, out, 400))
- ulunas_fp.c/h, ulunas_modules.c, ulunas_lut.c/h — 模型算子不变
- ulunas_matlab_weights.c/h — 权重不变
- fft_q15.h, qr_config.h, layer_dims.h — 不变
- Makefile — 路径更新为 linux_api12

## Build

```bash
make -C linux_api12 CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
```

## Deploy

```bash
ssh root@192.168.42.159 "killall ulunas_q15; sleep 1"
scp linux_api12/ulunas_q15 root@192.168.42.159:/data/tianhai/
ssh root@192.168.42.159 "cd /data/tianhai && nohup ./ulunas_q15 > /dev/null 2>&1 &"
```

## 待验证

- [ ] X2000 编译通过
- [ ] 实时运行 (RTF < 1.5)
- [ ] 降噪效果 vs linux_api11 baseline
- [ ] 跨设备稳定性 (与 WIN_INC=200 对比)
- [ ] 输出 RMS 校准 (OUTPUT_GAIN_Q15)
- [ ] Golden SNR vs MATLAB reference
