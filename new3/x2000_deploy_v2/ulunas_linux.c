/**
 * ulunas_linux.c — UL-UNAS Linux 降噪主程序
 * ==========================================
 * v8: 模型 16kHz 原生。8kHz 对讲机模式通过独立重采样器转换。
 *
 * 模式:
 *   measure      16kHz stdin→NR→16kHz stdout (文件测试)
 *   iccom        8kHz rf11→重采样16k→NR→重采样8k→wf12 (对讲机)
 *   test         8kHz stdin→AGC→NR+重采样→8kHz stdout
 *   bypass       rf11→wf12 直通
 *
 * 信号控制:
 *   kill -USR1 <pid>  切换降噪 ON/OFF
 *   kill -USR2 <pid>  切换 AGC ON/OFF
 */
#include "agc.h"
#include "resample_8k_16k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#define FRAME_IN_16K  400   /* 25ms @ 16kHz */
#define FRAME_IN_8K   200   /* 25ms @ 8kHz */

static volatile int g_noise_on = 1;
static volatile int g_agc_on   = 1;

static void sig_handler(int sig) {
    if (sig == SIGUSR1) {
        g_noise_on = !g_noise_on;
        fprintf(stderr, "\n[SIGUSR1] noise=%s\n", g_noise_on ? "ON" : "OFF");
    }
    if (sig == SIGUSR2) {
        g_agc_on = !g_agc_on;
        fprintf(stderr, "\n[SIGUSR2] agc=%s\n", g_agc_on ? "ON" : "OFF");
    }
}

static short resolve_flag(void) {
    if (g_noise_on && g_agc_on) return STATE_FLAG_BOTH;
    if (g_noise_on)             return STATE_FLAG_DENOISE;
    if (g_agc_on)               return STATE_FLAG_AGC;
    return STATE_FLAG_NONE;
}

int main(int argc, char **argv) {
    int test_mode    = (argc > 1 && strcmp(argv[1], "test") == 0);
    int bypass_mode  = (argc > 1 && strcmp(argv[1], "bypass") == 0);
    int measure_mode = (argc > 1 && strcmp(argv[1], "measure") == 0);

    if (measure_mode) {
        /* 16kHz native: model matches training rate, no resampling needed */
        fprintf(stderr, "UL-UNAS v8 — measure mode (16kHz stdin→NR→16kHz stdout, no AGC)\n");
        agc_init();
        short in[FRAME_IN_16K], out[FRAME_IN_16K]; int fid = 0;
        while (fread(in, sizeof(short), FRAME_IN_16K, stdin) == FRAME_IN_16K) {
            voice_NoiseReductionAndAGC(in, out, AGC_GOAL_HIGH, STATE_FLAG_DENOISE);
            fwrite(out, sizeof(short), FRAME_IN_16K, stdout); fflush(stdout); fid++;
        }
        fprintf(stderr, "measure done: %d frames\n", fid);

    } else if (test_mode) {
        /* 8kHz test with resampling: stdin 8k→16k→NR→8k→stdout */
        fprintf(stderr, "UL-UNAS v8 — test mode (8kHz stdin→resample→NR→resample→8kHz stdout)\n");
        agc_init();
        Upsampler8k16k up; upsampler_init(&up);
        Downsampler16k8k down; downsampler_init(&down);
        short in8k[FRAME_IN_8K], out8k[FRAME_IN_8K];
        short buf16k_in[FRAME_IN_16K], buf16k_out[FRAME_IN_16K];
        unsigned short energy_out = 0;

        while (fread(in8k, sizeof(short), FRAME_IN_8K, stdin) == FRAME_IN_8K) {
            /* 8k → 16k upsampling */
            upsampler_feed(&up, in8k, FRAME_IN_8K);
            while (upsampler_pop(&up, buf16k_in, FRAME_IN_16K) == FRAME_IN_16K) {
                /* AGC + NR at 16kHz */
                unsigned short cur = energy_calculate_and_smooth_s16(
                    buf16k_in, FRAME_IN_16K, ENERGY_HISTORY_FAC, &energy_out);
                int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
                voice_NoiseReductionAndAGC(buf16k_in, buf16k_out,
                    hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);

                /* 16k → 8k downsampling */
                downsampler_feed(&down, buf16k_out, FRAME_IN_16K);
            }
            /* Pop 8kHz output */
            int got = downsampler_pop(&down, out8k, FRAME_IN_8K);
            if (got > 0) {
                fwrite(out8k, sizeof(short), got, stdout);
                fflush(stdout);
            }
        }

    } else if (bypass_mode) {
        fprintf(stderr, "UL-UNAS v8 — BYPASS mode (rf11→wf12 passthrough)\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        #define FRAME_50MS_BP (FRAME_IN_8K * 2)
        short buf[FRAME_50MS_BP];
        while (1) { if (read(fd_rf, buf, sizeof(buf)) != sizeof(buf)) continue; write(fd_wf, buf, sizeof(buf)); }

    } else {
        /* iccom mode: 8kHz rf11 → resample → 16kHz NR → resample → 8kHz wf12 */
        signal(SIGUSR1, sig_handler);
        signal(SIGUSR2, sig_handler);
        fprintf(stderr, "UL-UNAS v8 — iccom mode (8kHz rf11→resample16k→NR→resample8k→wf12)\n");
        fprintf(stderr, "  kill -USR1 <pid>  toggle noise  kill -USR2 <pid>  toggle AGC\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }

        agc_init();
        Upsampler8k16k up; upsampler_init(&up);
        Downsampler16k8k down; downsampler_init(&down);
        fprintf(stderr, "UL-UNAS+AGC+Resample ready. noise=ON, agc=ON\n");

        #define FRAME_50MS_8K  (FRAME_IN_8K * 2)    /* 400 samples @ 8kHz = 50ms */
        #define FRAME_25MS_16K (FRAME_IN_16K)        /* 400 samples @ 16kHz = 25ms */
        short rf_in[FRAME_50MS_8K], wf_out[FRAME_50MS_8K];
        short buf16k_in[FRAME_25MS_16K], buf16k_out[FRAME_25MS_16K];
        unsigned short energy_out = 0;

        while (1) {
            ssize_t nr = read(fd_rf, rf_in, FRAME_50MS_8K * sizeof(short));
            if (nr != FRAME_50MS_8K * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n", nr, FRAME_50MS_8K * (int)sizeof(short));
                continue;
            }

            /* Feed 50ms of 8kHz → upsampler → 100ms of 16kHz (split into 4 × 25ms frames) */
            upsampler_feed(&up, rf_in, FRAME_50MS_8K);

            int wf_idx = 0;
            while (upsampler_pop(&up, buf16k_in, FRAME_25MS_16K) == FRAME_25MS_16K) {
                /* AGC + NR at 16kHz */
                unsigned short cur = energy_calculate_and_smooth_s16(
                    buf16k_in, FRAME_25MS_16K, ENERGY_HISTORY_FAC, &energy_out);
                int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
                short flag = resolve_flag();
                voice_NoiseReductionAndAGC(buf16k_in, buf16k_out,
                    hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW, flag);

                /* Feed 16kHz output → downsampler */
                downsampler_feed(&down, buf16k_out, FRAME_25MS_16K);
            }

            /* Pop 8kHz output → accumulate up to 50ms worth */
            int total_8k_out = 0;
            while (total_8k_out < FRAME_50MS_8K) {
                int got = downsampler_pop(&down, wf_out + total_8k_out,
                                          FRAME_50MS_8K - total_8k_out);
                if (got == 0) break;
                total_8k_out += got;
            }

            if (total_8k_out > 0)
                write(fd_wf, wf_out, total_8k_out * sizeof(short));
        }
        close(fd_rf); close(fd_wf);
    }
    return 0;
}
