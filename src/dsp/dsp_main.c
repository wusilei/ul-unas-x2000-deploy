/**
 * dsp_main.c — DM3730 DSP/BIOS 主程序 (UL-UNAS)
 * =============================================
 * 音频: McBSP 独占, ARM 不参与 PCM
 * 控制: DSPLink 短报文
 * 与 GTCRN 版区别:
 *   - noise_reduction(pcm_in, pcm_out, FRAME_SIZE) — 3 参数版本 (ulunas_nr.c)
 *   - 内部 WOLA + 三级后置增益
 *   - AGC 由 noise_reduction 内部调用 voice_NoiseReductionAndAGC
 */

#include <std.h>
#include <tsk.h>
#include <sem.h>
#include <string.h>

#include "dsp_mcbsp.h"
#include "dsp_link.h"
#include "noise_reduction.h"
#include "agc.h"

Void TSK_Audio(Void *arg) {
    int16_t pcm_in[FRAME_SIZE], pcm_out[FRAME_SIZE];
    for (;;) {
        dsp_mcbsp_read(pcm_in);
        if (g_dsp_state.denoise_enable) {
            /* UL-UNAS: noise_reduction 内部调用 voice_NoiseReductionAndAGC
             *         → WOLA + 三级后置增益 */
            noise_reduction(pcm_in, pcm_out, FRAME_SIZE);
        } else {
            memcpy(pcm_out, pcm_in, FRAME_SIZE * sizeof(int16_t));
        }
        dsp_mcbsp_write(pcm_out);
        g_dsp_state.frame_count++;
    }
}

Void TSK_Ctrl(Void *arg) {
    for (;;) { dsp_link_poll(); TSK_sleep(100); }
}

int main(void) {
    noise_init();
    dsp_mcbsp_init();
    dsp_link_init();
    g_dsp_state.denoise_enable = 1;
    g_dsp_state.agc_enable     = 1;
    return 0;
}
