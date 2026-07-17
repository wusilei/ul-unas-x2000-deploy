# UL-UNAS DM3730 DSP 移植

## 芯片: DM3730 C64x+ DSP @ 600MHz, L2 64KB

## 源文件说明

```
src/
├── model/                          ← X2000 移植 (不改动)
│   ├── ulunas_fp.c                 定點算子库                                53KB
│   ├── denoising_fp.h              定點精度定义 + 状态结构体                   17KB
│   ├── ulunas_infer.c              推理顶层                                  2KB
│   ├── ulunas_modules.c            Encoder/Decoder/GDPRNN 装配               63KB
│   ├── ulunas_lut.c                Sigmoid/Tanh/Log10/Sqrt LUT              22KB
│   ├── denoising_lut.h             LUT 声明                                  2KB
│   ├── ulunas_matlab_weights.c     409 张量权重                              965KB
│   ├── denoising_matlab_weights.h  权重 extern 声明                           27KB
│   ├── layer_dims.h                网络维度 (纯宏)                            9KB
│   └── qr_config.h                 逐层 Qr 移位参数 (纯宏)                    10KB
├── pipeline/
│   ├── ulunas_nr.c                 STFT→UL-UNAS→ISTFT→WOLA 端到端            14KB
│   └── noise_reduction.h
├── shared/
│   ├── fft_q15.h                   512-pt Q15 FFT (双向)                     10KB
│   ├── agc.c/h                     LMS AGC + 三级后置增益                    5KB
└── dsp/
    ├── dsp_mcbsp.c/h               McBSP + EDMA3 音频驱动
    ├── dsp_link.h                  DSPLink 控制协议
    ├── dsp_main.c                  DSP/BIOS 主循环 (UL-UNAS)
    ├── ulunas_config.tcf           DSP/BIOS 配置
    └── ulunas_link.cmd             DDR spill 配置 (16 COFF section)
```

## 编译

```bash
make            # 静态库 libulunas_dm6437.lib
make dsp        # 可执行文件 ulunas_dsp.out
make clean
```

## 编译器

- TI CGT C6000 v6.0.8
- `E:\app\CCStudio_v3.3\C6000\cgtools\bin\cl6x`
- Flags: `-mv64plus -O3 --opt_for_speed=4`

## v19 关键参数

| 参数 | 值 |
|------|-----|
| 采样率 | 8kHz |
| iccom 帧 | 200 点 (25ms) — v19 适配 DM6437 |
| STFT | WIN_LEN=512, WIN_INC=256, WOLA |
| 激活值 | s32f20 (Q20) |
| 权重 | 409 张量, 965KB, COFF 16 个 section |
| 浮点依赖 | 零 |
| 后置增益 | 三级 (近场1.0×, 中远场1.5×, 超远场0.25×) |

## 删除项 (vs X2000 原版)

- `ulunas_linux.c` → 替换为 `dsp_main.c`
- `fft_q15_msa.h` → MSA 版本 (DM6437 用标准 `fft_q15.h`)
- `test_*.c` → PC 测试代码
- iccom rf11/wf12 → McASP + EDMA3
- kill -USR1/USR2 → UART 串口协议
