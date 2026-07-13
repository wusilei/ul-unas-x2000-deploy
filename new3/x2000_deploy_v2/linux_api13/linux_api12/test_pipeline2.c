#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Expose internals for debugging
extern int g_frame_count;
extern int g_out_count;

int main() {
    noise_init();
    
    short in[400], out[400];
    for (int i = 0; i < 400; i++)
        in[i] = (short)(sinf(2.0f * 3.14159f * 1000.0f * i / 8000.0f) * 16000.0f);
    
    for (int call = 0; call < 15; call++) {
        noise_reduction(in, out, 400);
        
        // Check output
        long long sum_sq = 0;
        for (int i = 0; i < 400; i++) sum_sq += (int)out[i] * (int)out[i];
        double rms = sqrt(sum_sq / 400.0);
        printf("call=%2d frames=%3d out_fifo=%5d out_rms=%8.1f dBFS\n",
               call, g_frame_count, g_out_count, 20*log10(rms/32768.0));
    }
    noise_deinit();
    return 0;
}
