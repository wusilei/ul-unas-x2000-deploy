#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Mature project's log_gen_fixed + helpers
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_fp.h"

int main() {
    float stft_r[257], stft_i[257];
    FILE *f = fopen("dump_matlab/frame0_stft_real.bin", "rb");
    fread(stft_r, sizeof(float), 257, f); fclose(f);
    f = fopen("dump_matlab/frame0_stft_imag.bin", "rb");
    fread(stft_i, sizeof(float), 257, f); fclose(f);
    
    // Method 1: Inline float log10 (bm_test.c approach, 124 dB)
    int32_t x_log_float[257];
    for (int i = 0; i < 257; i++) {
        int32_t rq = (int32_t)round(stft_r[i] * 1048576.0);
        int32_t iq = (int32_t)round(stft_i[i] * 1048576.0);
        double re = rq / 1048576.0, im = iq / 1048576.0;
        double mag = sqrt(re*re + im*im);
        double clamped = (mag < 1e-12) ? 1e-12 : mag;
        x_log_float[i] = (int32_t)round(log10(clamped) * 1048576.0);
    }
    
    // Method 2: Mature integer sqrt + LUT (ulunas_fp.c, 84 dB)
    int32_t real_q20[257], imag_q20[257], x_log_int[257];
    for (int i = 0; i < 257; i++) {
        real_q20[i] = (int32_t)round(stft_r[i] * 1048576.0f);
        imag_q20[i] = (int32_t)round(stft_i[i] * 1048576.0f);
    }
    log_gen_fixed(real_q20, imag_q20, 257, x_log_int);
    
    // Compare
    double s = 0, e = 0;
    int max_diff = 0, max_bin = 0;
    for (int i = 0; i < 257; i++) {
        double g = x_log_float[i], c = x_log_int[i];
        s += g*g; e += (g-c)*(g-c);
        if (abs(x_log_float[i] - x_log_int[i]) > max_diff) {
            max_diff = abs(x_log_float[i] - x_log_int[i]);
            max_bin = i;
        }
    }
    printf("log_gen float vs int: SNR=%.2f dB, max_diff=%d at bin %d
", 10.0*log10(s/e), max_diff, max_bin);
    printf("  bin %d: float=%d int=%d
", max_bin, x_log_float[max_bin], x_log_int[max_bin]);
    printf("  bin 0: float=%d int=%d
", x_log_float[0], x_log_int[0]);
    printf("  bin 8: float=%d int=%d
", x_log_float[8], x_log_int[8]);
    return 0;
}
