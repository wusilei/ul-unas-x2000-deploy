# GTCRN 上/下采样管线分析

## 整体数据流

```
DSP 发 200×int16 @ 8kHz (每25ms)
       ↓
  [8k→16k 上采样]  ← 线性插值
       ↓
  [FIFO] → STFT (WIN_INC=256, WIN_LEN=512, Hann窗)
       ↓
  [GTCRN 推理] (Q15 FFT → denoise_infer_frame_q15)
       ↓
  [ISTFT] → OLA
       ↓
  [输出FIFO] → 16k→8k 降采样 (均值)
       ↓
DSP 收 200×int16 @ 8kHz
```

## 1. 8k→16k 上采样（线性插值）

```c
#define FRAME_IN    200   // 每次处理 200 个 8kHz 样本
#define FRAME_16K   400   // 产生 400 个 16kHz 样本
#define WIN_INC     256   // STFT 帧步进
#define WIN_LEN     512   // STFT 窗长

static int16_t g_last_in_8k;  // 上一帧的最后一个样本

for (int i = 0; i < FRAME_IN; i++) {
    int16_t s_curr = voiceIn[i];           // 当前 8k 样本
    int16_t s_prev = g_last_in_8k;         // 上一个 8k 样本

    // 样本1: 原始值
    g_fifo[g_fifo_wpos++] = s_curr;

    // 样本2: 线性插值中点 (prev + curr) / 2
    g_fifo[g_fifo_wpos++] = (int16_t)(((int)s_prev + (int)s_curr) >> 1);

    g_fifo_count += 2;
    g_last_in_8k = voiceIn[i];
}
```

**原理**：
```
8kHz:   S0 ─────── S1 ─────── S2 ─────── S3
          \      /   \      /   \      /
16kHz:    S0  mid01  S1  mid12  S2  mid23  S3
```

每两个 8k 样本之间插入一个平均值，200→400 样本，采样率翻倍。

**FRAME_IN=200 的由来**：
- DSP 每 50ms 发 400 个 8kHz 样本 → 但分两次调用 noise_reduction，每次 200
- 每次 200×8kHz → 400×16kHz → 400/256 = 1.5625 帧（不整除）
- **但 FIFO 缓冲吸收了不对齐**：win_len=512 > 256，有 256 点缓冲余量

## 2. STFT 处理（与下采样解耦）

```c
// FIFO 积累够 WIN_LEN=512 个 16kHz 样本后做一次 STFT
while (g_fifo_count >= WIN_LEN) {
    g_fifo_count -= WIN_INC;  // 消耗 256 个样本

    // 加 Hann 窗 → Q15 FFT → GTCRN 推理 → Q15 IFFT
    // ... (省略 STFT/ISTFT 细节，和 UL-UNAS 一致)

    // 合成输出写入 g_out_fifo
    for (int i = 0; i < WIN_INC; i++) {
        g_out_fifo[out_pos++] = clamp_q15(ola_sample);
        g_out_count++;
    }
}
```

**关键参数**：
- WIN_INC=256 → 帧率 = 16kHz/256 ≈ 62.5 Hz
- 每帧 257 bins (512-pt FFT)
- Hann 窗：`sin(π·i/511) × 32767` (Q15)

## 3. 16k→8k 降采样（均值）

```c
// 每 2 个 16kHz 样本平均 → 1 个 8kHz 样本
if (g_out_count >= FRAME_16K) {  // 400 个 16kHz 样本就绪

    // Warmup 淡入
    int32_t gain_q15 = 32768;  // 1.0
    if (g_frame_count < WARMUP_MUTE) gain_q15 = 0;           // 前20帧静音
    else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)       // 后12帧淡入
        gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;

    for (int i = 0; i < FRAME_IN; i++) {  // 输出 200 个 8kHz 样本
        int32_t a = (g_out_fifo[wr++] * gain_q15 + 16384) >> 15;
        int32_t b = (g_out_fifo[wr++] * gain_q15 + 16384) >> 15;
        voiceOut[i] = clamp_q15((a + b) >> 1);  // 均值降采样
    }
    g_out_count -= FRAME_16K;
}
```

**Warmup 参数**：
- WARMUP_MUTE=20 帧 → ~320ms 静音（等待 FIFO 填满）
- WARMUP_FADE=12 帧 → ~192ms 淡入
- 总计 ~512ms 预热

## 4. 为什么这方案跨设备稳定

| 特性 | GTCRN (WIN_INC=256 + 上采样) | UL-UNAS v6 (WIN_INC=200, 无上采样) |
|------|------------------------------|-------------------------------------|
| DSP 输入 | 200 样本/次 | 400 样本/次 |
| 帧对齐 | 上采样后 400→FIFO，有缓冲 | 400/200=2 帧精确对齐，无缓冲 |
| DSP 时序抖动容限 | 高（256点 FIFO 缓冲） | 低（必须精确 400/200=2） |
| 跨设备一致性 | ✅ 好 | ❌ 差（时序敏感） |

## 5. 移植到 UL-UNAS 的改动清单

### 5.1 ulunas_nr.c 改动

```c
// 修改前 (v6):
#define FRAME_IN    200
#define WIN_INC     200
// noise_reduction(in, out, n_samples) — 直接 8k 输入

// 修改后 (v12):
#define FRAME_IN    200       // 每次输入 200 个 8kHz 样本
#define FRAME_16K   400       // 上采样后 400 个 16kHz 样本
#define WIN_INC     256       // 恢复标准 WIN_INC
// 加入 g_last_in_8k、上采样循环、降采样循环
```

### 5.2 需要新增的变量
```c
static int16_t g_last_in_8k;   // 上一帧最后一个 8k 样本
```

### 5.3 需要修改的 WOLA 窗
- WIN_INC=256 → 窗函数是标准 2 窗重叠 Hann 窗
- 不需要 WIN_INC=200 的 3/2 混合窗 + WOLA 倒数表

### 5.4 需要修改的 Makefile / ulunas_linux.c
- 调用改为 `noise_reduction(in, out)` (GTCRN 风格，FRAME_IN=200 固定)
- 或保持 `noise_reduction(in, out, n_samples)` 接口不变

### 5.5 性能影响
- 上采样：200 次线性插值 → ~0.1ms
- 降采样：200 次均值 → ~0.1ms
- 总增加 ~0.2ms/frame，可忽略

## 6. 参考代码（直接可复用）

GTCRN 参考文件：
- `gtcrn-x2000-deploy/.../linux_api19_denoise/noise_reduction_q15.c` — 完整实现
- `gtcrn-x2000-deploy/.../linux_api19_denoise/fft_q15.h` — FFT（和 UL-UNAS 共享）

WIN_INC=256 的 Hann 窗生成：
```c
for (int i = 0; i < WIN_LEN; i++)
    hann_q15[i] = (int16_t)(sinf(3.14159265f * i / (WIN_LEN - 1)) * 32767.0f + 0.5f);
```
