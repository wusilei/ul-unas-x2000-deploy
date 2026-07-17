/**
 * dsp_mcbsp.h — DM3730 DSP McBSP + EDMA3 音频驱动
 * ==============================================
 * 8kHz, 16-bit, I2S 主模式, Ping-Pong 双缓冲
 * DSP 独占 McBSP, ARM 不参与 PCM, 不走 DSPLink
 */
#ifndef DSP_MCBSP_H
#define DSP_MCBSP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAMPLE_RATE     8000
#define FRAME_SIZE      200     /* 25ms @ 8kHz */
#define PING_PONG_SZ    FRAME_SIZE
#define RING_BUF_SZ     (FRAME_SIZE * 3)  /* 600 点环形缓存 */

/**
 * 初始化 McBSP0 (输入) + McBSP1 (输出) + EDMA3
 * 调用后音频自动开始流式采集/播放
 */
void dsp_mcbsp_init(void);

/**
 * 从环形缓存读取一帧 PCM Q15
 * 阻塞直到 FRAME_SIZE 点就绪, SCOM 事件驱动唤醒
 */
void dsp_mcbsp_read(int16_t *buf);

/**
 * 写入一帧 PCM Q15 到输出缓存
 * EDMA3 自动发送, 非阻塞
 */
void dsp_mcbsp_write(const int16_t *buf);

#ifdef __cplusplus
}
#endif

#endif /* DSP_MCBSP_H */
