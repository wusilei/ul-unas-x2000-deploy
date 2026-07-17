/**
 * dsp_link.h — DM3730 DSPLink 控制协议
 * =====================================
 * 代替原 X2000 的 kill -USR1/USR2 信号
 * 仅承载短报文 (控制/告警/打印), 禁止音频 PCM
 */
#ifndef DSP_LINK_H
#define DSP_LINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LINK_SYNC0      0xAA
#define LINK_SYNC1      0x55
#define LINK_FRAME_LEN  8

enum {
    CMD_DENOISE_ONOFF  = 0x01,
    CMD_AGC_ONOFF      = 0x02,
    CMD_AGC_THRESHOLD  = 0x03,
    CMD_GET_STATUS     = 0x04,
};

typedef struct {
    uint8_t  denoise_enable;
    uint8_t  agc_enable;
    uint8_t  reserved[2];
    uint16_t frame_count;
    uint16_t overload_count;
    uint16_t cpu_load;
} DspState;

extern volatile DspState g_dsp_state;

void dsp_link_init(void);
void dsp_link_poll(void);       /* DSPLink MSGQ 非阻塞收, 100ms 周期 */

#ifdef __cplusplus
}
#endif

#endif /* DSP_LINK_H */
