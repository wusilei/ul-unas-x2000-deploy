/**
 * denoise_linux.c — DENOISE Linux 侧降噪主程序 (v21 — linux_api19_denoise)
 * =====================================================================
 * 信号控制:
 *   kill -USR1 <pid>  切换降噪 ON/OFF
 *   kill -USR2 <pid>  切换 AGC ON/OFF
 */
#include "agc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#define FRAME_IN   200

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
        fprintf(stderr, "UL-UNAS v2 — measure mode (8kHz stdin→NR→8kHz stdout, no AGC)\n");
        agc_init();
        #define FRAME_MEASURE 400
        short in[FRAME_MEASURE], out[FRAME_MEASURE]; int fid = 0;
        while (fread(in, sizeof(short), FRAME_MEASURE, stdin) == FRAME_MEASURE) {
            voice_NoiseReductionAndAGC(in, out, AGC_GOAL_HIGH, STATE_FLAG_DENOISE, FRAME_MEASURE);
            fwrite(out, sizeof(short), FRAME_MEASURE, stdout); fflush(stdout); fid++;
        }
        fprintf(stderr, "measure done: %d frames\n", fid);
    } else if (test_mode) {
        fprintf(stderr, "UL-UNAS v2 — test mode (8kHz stdin→AGC→NR→8kHz stdout)\n");
        agc_init();
        #define FRAME_TEST 400
        short in[FRAME_TEST], out[FRAME_TEST];
        unsigned short energy_out = 0;
        while (fread(in, sizeof(short), FRAME_TEST, stdin) == FRAME_TEST) {
            unsigned short cur = energy_calculate_and_smooth_s16(
                in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            long long goal = hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW;
            voice_NoiseReductionAndAGC(in, out, goal, STATE_FLAG_BOTH, FRAME_TEST);
            fwrite(out, sizeof(short), FRAME_TEST, stdout); fflush(stdout);
        }
    } else if (bypass_mode) {
        fprintf(stderr, "UL-UNAS v2 — BYPASS mode (rf11→wf12 passthrough)\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        #define FRAME_50MS_BP (FRAME_IN * 2)
        short buf[FRAME_50MS_BP];
        while (1) { if (read(fd_rf, buf, sizeof(buf)) != sizeof(buf)) continue; write(fd_wf, buf, sizeof(buf)); }
    } else {
        signal(SIGUSR1, sig_handler);
        signal(SIGUSR2, sig_handler);
        fprintf(stderr, "UL-UNAS v2 — iccom mode (rf11→AGC→NR→wf12, 400smp/50ms)\n");
        fprintf(stderr, "  kill -USR1 <pid>  toggle noise  kill -USR2 <pid>  toggle AGC\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }

        agc_init();
        fprintf(stderr, "UL-UNAS+AGC ready. noise=ON, agc=ON\n");

        #define FRAME_50MS (FRAME_IN * 2)
        short in[FRAME_50MS], out[FRAME_50MS];
        unsigned short energy_out = 0;

        while (1) {
            ssize_t nr = read(fd_rf, in, FRAME_50MS * sizeof(short));
            if (nr != FRAME_50MS * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n", nr, FRAME_50MS * (int)sizeof(short));
                continue;
            }

            /* 能量检测 (用前200样本决定goal, 与GTCRN一致) */
            unsigned short cur = energy_calculate_and_smooth_s16(in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            short flag = resolve_flag();
            voice_NoiseReductionAndAGC(in, out, hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW, flag, FRAME_50MS);

            write(fd_wf, out, FRAME_50MS * sizeof(short));
        }
        close(fd_rf); close(fd_wf);
    }
    return 0;
}
