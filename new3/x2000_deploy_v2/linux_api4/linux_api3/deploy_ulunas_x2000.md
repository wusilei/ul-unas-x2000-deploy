# UL-UNAS 部署 — X2000

**文件**: `ulunas_q15`（一个二进制文件，无依赖，~1.1MB MIPS32R2 静态链接）

---

## Linux 侧部署

```bash
# 1. 传文件
scp ulunas_q15 root@192.168.42.159:/data/

# 2. 连接
ssh root@192.168.42.159

# 3. 启动
/data/ulunas_q15 &
```

## 模型启动
```
看到 UL-UNAS+AGC ready. noise=ON, agc=ON 即就绪。
看到 rf11 short read: 768/800  模型正式启动（表示 DSP 已开始发数据）
```

## DSP 侧对接

DSP 工程师需要确保以下两点：

### 1. iccom 通道

| 通道 | 方向 | 数据格式 | 周期 |
|:--|:--|:--|:--|
| `/dev/iccom_rf11` | DSP → Linux | 400×int16 PCM @ 8kHz | 每 50ms 发一包 |
| `/dev/iccom_wf12` | Linux → DSP | 400×int16 PCM @ 8kHz | 每 50ms 收一包 |

DSP 侧需要：
- 打开 `/dev/iccom_rf11` 写，每 50ms 写入 400 个 int16（800 bytes）
- 打开 `/dev/iccom_wf12` 读，每 50ms 读取 400 个 int16（800 bytes）
- Linux 侧会将 400 samples 拆成 2×200 逐帧处理后再写回

### 2. 开机自启顺序

DSP 固件需要在 iccom 设备创建之后再启动 Linux 侧进程。`/data/tianhai/run.sh` 已配置等待 iccom 就绪后启动。

```bash
# run.sh 中的启动逻辑（已配好）
if [ -x /data/ulunas_q15 ]; 
then
    for i in 1 2 3 4 5 6 7 8 9 10; do
        [ -c /dev/iccom_rf11 ] && [ -c /dev/iccom_wf12 ] && break
        sleep 1
    done
    /data/ulunas_q15 &
fi
```

## iccom 数据流

```
RTOS (DSP 侧)                    Linux 侧
─────────────                    ────────
每 50ms: write(rf11, 400×int16)  →  read(rf11)
                                     → 8k→16k 上采样
                                     → Q15 FFT (纯定点)
                                     → UL-UNAS 推理 (全定点)
                                     → Q15 IFFT + WOLA
                                     → 16k→8k 下采样
                                  ←  write(wf12, 400×int16)
每 50ms: read(wf12)
```

**通道约定**: `rf11` (RTOS→Linux) / `wf12` (Linux→RTOS)，400×int16 @ 8kHz，50ms/帧。

## 管线架构

```
8kHz PCM → AGC (int32) → 16kHz upsample → Q15 Hann window
→ Q15 FFT (fft_q15_forward) → float cast → UL-UNAS infer
→ Q20 CRM output → >>5 → Q15 → Q15 IFFT (fft_q15_inverse)
→ Hann synthesis (>>20) → OLA → 16k→8k downsample → 8kHz PCM
```

全部定点运算（仅 FFT→UL-UNAS 接口处有一次 float 转换，257 bins × 1 次除法）。

## X2000 性能

| 指标 | 值 |
|------|-----|
| 内存 | 800 KB RSS |
| 推理延迟 | ~32 ms/帧 (200smp @ 8kHz) |
| 实时因子 | ~1.27×（近实时） |
| 二进制大小 | 1.1 MB |

## Linux 侧常用

```bash
# 查看进程
ps | grep ulunas

# 切换降噪 ON/OFF
kill -USR1 $(ps | grep ulunas_q15 | grep -v grep | awk '{print $1}')

# 切换 AGC ON/OFF
kill -USR2 $(ps | grep ulunas_q15 | grep -v grep | awk '{print $1}')

# 停止进程
kill $(ps | grep ulunas_q15 | grep -v grep | awk '{print $1}')

# 旁路测试（直通，不降噪）
killall ulunas_q15
/data/ulunas_q15 bypass &
```
