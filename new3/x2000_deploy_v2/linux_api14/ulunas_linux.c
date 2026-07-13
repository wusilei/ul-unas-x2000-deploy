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
        fprintf(stderr, "UL-UNAS v13 — iccom mode + elastic buffer (rf11→buf→AGC→NR→wf12)\n");
        fprintf(stderr, "  kill -USR1 <pid>  toggle noise  kill -USR2 <pid>  toggle AGC\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }

        agc_init();
        fprintf(stderr, "UL-UNAS+AGC ready. noise=ON, agc=ON\n");

        #define FRAME_50MS (FRAME_IN * 2)
        #define RBUF_SZ   1200  /* elastic ring buffer: 3× iccom frame */
        short rbuf[RBUF_SZ];
        int   rbuf_wpos = 0, rbuf_count = 0;
        short in[FRAME_50MS], out[FRAME_50MS];
        unsigned short energy_out = 0;

        while (1) {
            /* Read whatever DSP sent into ring buffer */
            short tmp[FRAME_50MS];
            ssize_t nr = read(fd_rf, tmp, sizeof(tmp));
            if (nr > 0) {
                int n_samples = nr / sizeof(short);
                for (int i = 0; i < n_samples; i++) {
                    rbuf[rbuf_wpos] = tmp[i];
                    rbuf_wpos = (rbuf_wpos + 1) % RBUF_SZ;
                    rbuf_count++;
                }
            }

            /* Process while enough samples in buffer */
            while (rbuf_count >= FRAME_50MS) {
                /* Pull 400 samples from ring buffer */
                int rpos = (rbuf_wpos - rbuf_count + RBUF_SZ) % RBUF_SZ;
                for (int i = 0; i < FRAME_50MS; i++) {
                    in[i] = rbuf[(rpos + i) % RBUF_SZ];
                }
                rbuf_count -= FRAME_50MS;

                /* Energy detection + AGC gate (same as api11) */
                unsigned short raw_energy = energy_calculate_and_smooth_s16(
                    in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
                long long goal; int gn, gd;
                agc_gate_decision(raw_energy, &goal, &gn, &gd);
                short flag = resolve_flag();
                if (flag == STATE_FLAG_BOTH && raw_energy > ENERGY_FAR_THR) {
                    for (int i = 0; i < FRAME_50MS; i++) {
                        int32_t v = (int32_t)in[i] * 3 / 2;
                        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                        in[i] = (short)v;
                    }
                }
                voice_NoiseReductionAndAGC(in, out, goal, flag, FRAME_50MS);
                write(fd_wf, out, FRAME_50MS * sizeof(short));
            }
        }
        close(fd_rf); close(fd_wf);
    }
    return 0;
}
