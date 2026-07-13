# linux_api13 -- 16kHz 内部处理，带抗混叠滤波器 (2026-07-10)

## 当前 X2000 状态
- IP: 192.168.42.159
- 运行: **linux_api13** (已部署验证)
- 上一版本: linux_api12 (8k→16k 线性插值, 无抗混叠)

## 核心变更 vs linux_api12

### 问题
linux_api12 使用纯线性插值(2x)和均值降采样(2x)，缺少抗混叠/抗镜像滤波器:
- 镜像频率混入 4-8kHz 频段
- 降采样前 4-8kHz 的残余分量折返到 0-4kHz
- 高频失真 + 混叠噪声

### 方案
在上下采样管线中插入 16 阶对称 FIR 低通滤波器 (fc=4kHz@16kHz):

`
8k PCM in -> [linear interp 2x] -> [16-tap LPF, fc=4kHz] -> STFT(512,256)@16kHz
  -> NR -> WOLA -> [16-tap LPF, fc=4kHz] -> [decimate 2x] -> 8k PCM out
`

### 滤波器设计
- 类型: 16 阶对称 FIR, Hamming 窗
- 截止: fc=4kHz @ Fs=16kHz (归一化 0.5)
- 系数: Q15 定点
  `
  -79, -136, 312, 654, -1244, -2280, 4501, 14655,
  14655, 4501, -2280, -1244, 654, 312, -136, -79
  `
- 直流增益: ~1.0 (Q15 归一化)
- 流式状态: g_up_lpf_hist[15] / g_dn_lpf_hist[15] 保持块间连续性

### 其他不变项
- NR 内核: Q15 FFT + UL-UNAS + WOLA, 与 linux_api12 完全相同
- 帧参数: N_FFT=512, WIN_LEN=512, WIN_INC=256 (16ms@16kHz)
- AGC: 独立模块, 在 NR 之前/之后可选
- 信号控制: kill -USR1 (降噪 ON/OFF), kill -USR2 (AGC ON/OFF)
- 帧对齐: 输入 400 samples (50ms@8kHz), 内部 800 samples (50ms@16kHz)
- 输入: /dev/iccom_rf11, 输出: /dev/iccom_wf12

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入采样率 | 8000 Hz | iccom 接口 |
| 内部采样率 | 16000 Hz | 上采样后 |
| WIN_LEN | 512 | 32ms @ 16kHz |
| WIN_INC | 256 | 16ms @ 16kHz |
| N_BINS | 257 | STFT bins |
| 上采样方式 | 线性插值 + 16-tap FIR LPF | 抗镜像 |
| 降采样方式 | 16-tap FIR LPF + 2:1 抽取 | 抗混叠 |
| WOLA | Hann 窗, power-complementary, 无需归一化 |
| 定点格式 | 外部位 Q15, 推理 Q20, FFT Q15 |
| 帧延迟 | ~32ms (WIN_LEN) + ~8ms (LPF group delay) |

## 交叉编译 & 部署

### VM 信息
- 名称: ubuntu-20.04 (VirtualBox)
- IP: 192.168.56.101 (Host-Only)
- 用户/密码: a / a
- 共享目录: /media/sf_haidesi/ → D:\haidesi\

### 编译
`ash
# 方式 1: VM 内直接编译
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/linux_api13
make clean all

# 方式 2: 通过 VBoxManage (Windows)
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' guestcontrol "ubuntu-20.04" run \
  --username a --password "a" --exe /usr/bin/make \
  -- -C /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/linux_api13 clean all

# 跨编译器: /home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc (GCC 7.2.0)
# CFLAGS: -O3 -std=c99 -march=mips32r2 -msoft-float
# 链接: -static (完全静态, 无依赖)
`

### 部署
`ash
# VM 内部署 (make deploy):
make deploy

# 或手动:
ssh root@192.168.42.159 "killall ulunas_q15 2>/dev/null; sleep 1"
scp ulunas_q15 root@192.168.42.159:/data/
ssh root@192.168.42.159 "cp /data/ulunas_q15 /data/tianhai/ && chmod +x /data/ulunas_q15"
ssh root@192.168.42.159 "cd /data && nohup ./ulunas_q15 > /dev/null 2>&1 &"
`

### 验证
`ash
ssh root@192.168.42.159 "ps | grep ulunas_q15"
# 预期: ./ulunas_q15 进程, CPU 约 80-90%

# PC 测试模式 (用于音频质量评估):
make pc_test
./pc_test_nr test < input.pcm > output.pcm  # 8kHz, 16-bit, mono
`

### 信号控制 (运行时)
`ash
kill -USR1 <pid>   # 切换降噪 ON/OFF
kill -USR2 <pid>   # 切换 AGC ON/OFF
`

## 已通过验证
- [x] 交叉编译通过 (MIPS gcc 7.2.0)
- [x] X2000 部署并运行
- [x] 内存: 1244 KB (VmSize), 1080 KB (VmRSS)
- [x] CPU: ~83% (单核 MIPS)
- [x] iccom 接口: rf11→wf12 数据流通

## 待验证
- [ ] 降噪效果 vs linux_api12 (主观听感)
- [ ] 输出 RMS 校准
- [ ] Golden SNR 测量
- [ ] 长时间运行稳定性 (内存泄漏检查)
- [ ] 上下采样边界不连续检查 (LPF 流式状态)
- [ ] 跨设备一致性

## 程序启动模式
`ash
ulunas_q15              # iccom 模式: rf11→AGC→NR→wf12 (生产)
ulunas_q15 test         # stdin→AGC→NR→stdout (8kHz PCM)
ulunas_q15 bypass       # rf11→wf12 直通
ulunas_q15 measure      # stdin→NR→stdout (无 AGC)
`

## 二进制
- 文件: ulunas_q15
- 格式: ELF 32-bit MIPS, MIPS32 rel2, statically linked
- 大小: ~1090 KB
- 栈: 132 KB
- 数据段: 180 KB
