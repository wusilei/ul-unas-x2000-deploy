/**
 * nn_iccom.h — X2000 iccom Communication + Main Loop Template
 * ============================================================
 * Template for X2000 Linux-side iccom denoise daemon.
 *
 * iccom channels: rf11 (DSP→Linux, read), wf12 (Linux→DSP, write)
 * Data format: 400×int16 PCM @ 8kHz, 50ms per frame
 *
 * Signal handling:
 *   SIGUSR1: toggle denoise ON/OFF
 *   SIGUSR2: toggle AGC ON/OFF
 *
 * Typical main.c:
 *   #include "nn_infer.h"
 *   int main(void) { return nn_iccom_main(400, my_infer_cb, &my_state); }
 *
 * Verified: GTCRN (denoise_v19_q15) + UL-UNAS (linux_api9)
 * License: MIT
 */

#ifndef NN_ICCOM_H
#define NN_ICCOM_H

#include "nn_core/nn_qformat.h"
#include "nn_pipeline/nn_agc.h"
#include "nn_pipeline/nn_pipeline.h"
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Global flags for USR1/USR2 toggling */
static volatile int nn_iccom_noise_on = 1;
static volatile int nn_iccom_agc_on = 1;

static void nn_iccom_sigusr1(int sig) { (void)sig; nn_iccom_noise_on = !nn_iccom_noise_on; }
static void nn_iccom_sigusr2(int sig) { (void)sig; nn_iccom_agc_on = !nn_iccom_agc_on; }

/**
 * nn_iccom_main — X2000 iccom main loop (BLOCKING)
 *
 * n_samples: PCM samples per frame (400 for 50ms @ 8kHz)
 * infer_cb:  model inference callback
 * cb_user:   user data for callback
 * agc_state: NULL or pointer to nn_agc_state_t for AGC (optional)
 * pipeline:  NULL or pre-created nn_pipeline_t (NULL = auto-create with ULUNAS config)
 *
 * Returns: never (infinite loop), or -1 on fatal error
 */
static inline int nn_iccom_main(int n_samples,
                                 void (*infer_cb)(const int32_t*, const int32_t*, int, int32_t*, void*),
                                 void *cb_user,
                                 void *agc_state,   /* nn_agc_state_t* */
                                 void *pipeline)    /* nn_pipeline_t* */
{
    /* Signal setup */
    signal(SIGUSR1, nn_iccom_sigusr1);
    signal(SIGUSR2, nn_iccom_sigusr2);
    signal(SIGPIPE, SIG_IGN);

    /* Open iccom devices */
    int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
    int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
    if (fd_rf < 0 || fd_wf < 0) {
        fprintf(stderr, "FATAL: cannot open iccom devices\n");
        if (fd_rf >= 0) close(fd_rf);
        return -1;
    }

    /* Allocate PCM buffers */
    int16_t *pcm_in = (int16_t*)malloc(n_samples * sizeof(int16_t));
    int16_t *pcm_out = (int16_t*)malloc(n_samples * sizeof(int16_t));
    if (!pcm_in || !pcm_out) { free(pcm_in); free(pcm_out); close(fd_rf); close(fd_wf); return -1; }

    fprintf(stderr, "NN denoise ready. noise=ON, agc=ON\n");

    while (1) {
        /* Read from DSP */
        int nr = 0;
        while (nr < n_samples * (int)sizeof(int16_t)) {
            int n = read(fd_rf, (char*)pcm_in + nr, n_samples * sizeof(int16_t) - nr);
            if (n <= 0) { usleep(1000); continue; }
            nr += n;
        }

        /* Optional AGC */
        if (agc_state && nn_iccom_agc_on) {
            nn_agc_process((nn_agc_state_t*)agc_state, pcm_in, pcm_out, n_samples);
            /* Swap */
            int16_t *tmp = pcm_in; pcm_in = pcm_out; pcm_out = tmp;
        }

        /* Pipeline process or direct callback */
        if (pipeline) {
            int n_out = 0;
            nn_pipeline_process((nn_pipeline_t*)pipeline, pcm_in, n_samples,
                                (nn_infer_callback_t)infer_cb, cb_user, pcm_out, &n_out);
        } else {
            /* Direct: user handles STFT/ISTFT themselves */
            if (nn_iccom_noise_on)
                infer_cb(NULL, NULL, 0, NULL, cb_user);  /* passthrough signal */
            else
                memcpy(pcm_out, pcm_in, n_samples * sizeof(int16_t));
        }

        /* Write to DSP */
        if (nn_iccom_noise_on || !pipeline) {
            int nw = 0;
            while (nw < n_samples * (int)sizeof(int16_t)) {
                int n = write(fd_wf, (char*)pcm_out + nw, n_samples * sizeof(int16_t) - nw);
                if (n <= 0) { usleep(1000); continue; }
                nw += n;
            }
        } else {
            /* Bypass mode: passthrough input */
            int nw = 0;
            while (nw < n_samples * (int)sizeof(int16_t)) {
                int n = write(fd_wf, (char*)pcm_in + nw, n_samples * sizeof(int16_t) - nw);
                if (n <= 0) { usleep(1000); continue; }
                nw += n;
            }
        }
    }

    /* Unreachable */
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* NN_ICCOM_H */
