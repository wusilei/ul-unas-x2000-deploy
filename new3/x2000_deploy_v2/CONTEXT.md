
## linux_api13 — 16kHz 内部处理 (WIP, 2026-07-08)

**设计**：保持 N_FFT=512 / WIN_INC=256 / ERB 权重不变，仅在 I/O 侧加入线性插值 2x 上下采样。

```
DSP 400xint16@8k → linear interp(2x) → 800xint16@16k → FIFO
→ STFT(N_FFT=512,WIN_INC=256) → 推理 → ISTFT → WOLA
→ decimate(2x): 800@16k → 400@8k → DSP
```

**与 v12 的差异**（只改了 ulunas_nr.c）：
- 输入：400 → 线性插值 2x → 800，推入 FIFO
- 推理管线：完全不变（N_FFT=512, WIN_INC=256, WOLA 表不变）
- 输出：从 FIFO 拉 800，抽取 2x（只取偶数下标），写回 400

**收敛性**：LCM(800,256)=12800，16 调用=50 帧，周期性有界振荡，输出 FIFO 不会欠载。
帧模式：3,3,3,3,3,3,3,4,3,3,3,3,3,3,3,4（每 16 调用）
FIFO 水位：+32/调用(3帧) -224(4帧)，稳态振幅 ±224，水位 ≥320，FIFO_SZ=2048 充裕。

**状态**：代码已写完，待交叉编译 + X2000 验证。

**编译部署**（在有交叉工具链的 Linux 上执行）：
```bash
cd linux_api13
make        # 交叉编译
make deploy # 上传 X2000
```

或在 X2000 上直接编译：
```bash
scp linux_api13/* root@192.168.42.159:/tmp/api13/
ssh root@192.168.42.159 "cd /tmp/api13 && gcc -O2 -std=c99 -o /data/ulunas_q15 agc.c ulunas_linux.c ulunas_fp.c ulunas_lut.c ulunas_modules.c ulunas_matlab_weights.c ulunas_infer.c ulunas_nr.c -lm && cp /data/ulunas_q15 /data/tianhai/ && killall ulunas_q15; sleep 1; nohup /data/ulunas_q15 &"
```

**预期结果**（参考此前 PC 端 16kHz 版本测试）：
- 衰减 ~-4.1dB（比 8kHz 直通的 -1.1dB 更大，符合预期：16kHz 模型能看到 4-8kHz 噪声并抑制）
- 无延迟、无失真（如果 X2000 实机通过）

### 已知风险
- **MIPS**：16kHz 内部处理 → 每调用平均 3.125 帧（vs 8kHz 的 1.5625 帧），推理量翻倍。v12 RTF=1.27x，v13 预计 RTF≈2.5x（可能超标过 1.0）。**如果不能过 1.0，最直接的后手是退回 v12。**
- 输出 FIFO 稳态水位于 320-544 之间，足够安全。但如果 gcc -O0 或 MIPS 性能不足导致帧处理跟不上输入，FIFO 会持续增长最终爆满——逻辑上不会数据损坏，但延迟会积累。


# UL-UNAS X2000 定点部署 — 会话上下文 (2026-07-08)

## 项目概述
UL-UNAS 语音增强模型 → 纯定点 C 推理 → 君正 X2000 (MIPS32R2, 无 FPU, 192.168.42.159)
MATLAB 源码: `UL-UNAS_SE_FPversion_v2/` | C 代码: `c_version/x2000_deploy_v2/`

## 推理管线
```
8kHz PCM → AGC → 16kHz upsample → Q15 Hann window → Q15 FFT (fft_q15_forward)
→ Q15<<5=Q20 直通 → UL-UNAS infer (log_gen→BM→Encoder→GDPRNN×2→Decoder→BS→MASK)
→ Q20>>5=Q15 → Q15 IFFT (fft_q15_inverse) → WOLA (window²归一化) → 16k→8k downsample
```

## X2000 部署 (linux_api/)
```
linux_api/
├── Makefile              # make / make deploy
├── deploy_ulunas_x2000.md # 部署文档 (iccom通道、信号控制)
├── FREEZE.md
├── ulunas_q15            # MIPS32R2 静态二进制 (~1MB)
├── ulunas_linux.c        # iccom 主循环 (rf11→wf12, 400×int16 @ 50ms)
├── ulunas_nr.c           # 全定点降噪: Q15 FFT + WOLA + UL-UNAS (272行, v3)
├── ulunas_fp.c/h         # 模型算子
├── ulunas_infer.c        # ulunas_infer_frame() 推理封装
├── ulunas_lut.c/h        # sigmoid/tanh/log10/sqrt LUT (log10_q20 已修复)
├── ulunas_modules.c      # Encoder/Decoder/GDPRNN 模块
├── ulunas_matlab_weights.c/h  # 409个权重数组 (~944KB)
├── agc.c/h               # voice_AGC LMS
├── fft_q15.h             # Q15 512-pt FFT (forward+inverse, header-only)
├── noise_reduction.h     # noise_reduction() 接口
├── qr_config.h           # 全部 Qr 移位参数
└── layer_dims.h          # 网络维度
```
编译: `make` | 部署: `make deploy` | ssh root@192.168.42.159
X2000: `/data/ulunas_q15` | 自启: `/data/tianhai/run.sh`
切换降噪: `kill -USR1 <pid>` | 切换AGC: `kill -USR2 <pid>`

## 全部 Bug 修复 (10+ 个, git log)
1. state_init 移出帧循环 → E0.tconv: 1.4→110 dB
2. gru_module sum-then-round → 匹配 MATLAB 累加顺序
3. Golden 比对 col→row-major (FA GRU)
4. pconv2d_func weight_stride → 全部分组 PConv +11~13 dB
5. BN weight uint16 饱和 → E1.pconv0: 16→96 dB
6. E1/E2/E3/E4/D0/D1/D3 TA FC Qr 修正 (MATLAB 2^(-9)~2^(-10))
7. ln_func channel index: c=i/W → c=i%C ([T][C] row-major)
8. inter_rnn_module 每步独立隐状态 (消除伪序列依赖)
9. D0/D1/D3 Decoder FA GRU Qr (-13/-8→-12/-7, -11/-6)
10. non_gtconv2d_func kernel reversal (MATLAB rot90)
11. log10_q20 LUT: exponent=log2-19, Q20直接相加 (不用除法)

## PC Golden 测试状态 (Frame 0, ./test_matlab_golden)
| 层 | SNR | 状态 |
|----|-----|------|
| BM | 84 dB | PASS (定点 log_gen 牺牲 40dB 但对降噪影响小) |
| E0.tconv | 81 dB | PASS |
| E1.pconv0..ctfa_in | 94-109 dB | PASS |
| D0.pconv0..tconv | 105-109 dB | PASS |
| D3.pconv0..tconv | 93-98 dB | PASS |
| E0.ta | 51 dB | WARN |
| E0.fa / RNN intra_gru | 38 dB | BiGRU nHidden=4 精度天花板 |
| E1-E4 iso, D0-D4 iso | 42-61 dB | WARN |
| RNN chain | 6-8 dB | golden版本不匹配 (RNN golden未刷新) |
| Dec chain / MASK | 7-13 / -13~1 dB | RNN误差传播 |

## X2000 实时性能
- 内存: 800KB RSS | 延迟: ~32ms/帧 | RTF: ~1.27×
- 全定点 (零 float, 零 int64 除法) — fft_q15.h + Q15<<5 直通

## ⚠️ 问题定位 (2026-07-08 确认)
**libdenoise.so + numpy STFT = 完美** (session6: RMS降噪 6.6dB, 无失真, 无延迟)
**ulunas_q15 + fft_q15 + WOLA + AGC = 差** (人声失真, 噪声未消, 3秒延迟)

→ **UL-UNAS 模型推理完全正确。问题100%在 STFT/ISTFT/WOLA/AGC 前端。**

## 根因分析: Python vs X2000 pipeline 差异
| 环节 | Python (正确, 已验证) | X2000 (ulunas_nr.c v3, 有问题) |
|------|----------------------|-------------------------------|
| 分析/合成窗 | numpy FFT (无1/N, 自动) | fft_q15 (内部缩放未知, 需校准) |
| 推理输入 | ctypes float (自动Q转换) | Q15<<5=Q20 直通 (零float, 缩放正确) |
| OLA | numpy IFFT/WOLA (成熟库) | 手写 WOLA + 倒数查表 (可能有bug) |
| 上采样 | 8k→16k 线性插值 (同C) | 8k→16k 线性插值 (相同逻辑) |
| 下采样 | 16k→8k 平均对 (同C) | 16k→8k 平均对 (相同逻辑) |
| AGC | 未使用 | agc.c voice_AGC (可能引入失真) |
| 缓冲区 | 无 (全帧处理) | FIFO + OLA 缓冲 (延迟3s根因) |

## 下一步调试 (按优先级)
1. **PC端 A/B 对比**: 用同一16kHz音频文件跑 Python numpy pipeline 和 C fft_q15 pipeline, 比较 PCM 输出
2. **fft_q15 缩放校准**: 确认总增益 unity (目前 >>24 合成, 但 Python 的 IFFT 自动除 N)
3. **旁路 AGC**: 测试 STATE_FLAG_DENOISE 模式排除 AGC 影响
4. **WOLA 归一化验证**: 检查 window² 叠加和倒数表是否正确
5. **上/下采样验证**: 16k↔8k 线性插值是否引入混叠/丢高频
6. **FIFO/OLA 缓冲重构**: 消除 3s 延迟 (考虑全帧批处理替代滑动窗口实时处理)

## 关键文件路径
C代码: `/media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/`
部署: `linux_api/` 子目录 | Golden: `dump_matlab/` (445文件, gitignored)
测试音频: `../test_wavs/session5.wav` (8kHz对讲机实录, 42s)
Python参考: `denoise_wav.py` (ctypes调用libdenoise.so + numpy STFT)
测试: `make pc && ./test_matlab_golden ./dump_matlab/`

