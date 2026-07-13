// Quick pipeline test: feed sine wave, check if output is non-zero
#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    noise_init();
    
    // Generate 400-sample 8kHz sine wave (1kHz tone, Q15)
    short in[400];
    for (int i = 0; i < 400; i++) {
        in[i] = (short)(sinf(2.0f * 3.14159f * 1000.0f * i / 8000.0f) * 16000.0f);
    }
    
    short out[400];
    memset(out, 0xCD, sizeof(out)); // Fill with garbage
    
    // Run 10 frames to get past warmup
    for (int f = 0; f < 10; f++) {
        noise_reduction(in, out, 400);
    }
    
    // Check output
    long long sum_sq = 0;
    int non_zero = 0;
    for (int i = 0; i < 400; i++) {
        sum_sq += (int)out[i] * (int)out[i];
        if (out[i] != 0) non_zero++;
    }
    
    double rms = sqrt(sum_sq / 400.0);
    printf("Output: non_zero=%d/400, RMS=%.1f (Q15), RMS_dB=%.1f dBFS\n",
           non_zero, rms, 20*log10(rms/32768.0));
    
    noise_deinit();
    return 0;
}
