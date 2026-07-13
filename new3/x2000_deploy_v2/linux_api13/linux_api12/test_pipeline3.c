#include "noise_reduction.h"
#include <stdio.h>
#include <math.h>

int main() {
    noise_init();
    short in[400], out[400];
    for (int i = 0; i < 400; i++)
        in[i] = (short)(sinf(2.0f * 3.14159f * 1000.0f * i / 8000.0f) * 16000.0f);
    
    for (int call = 0; call < 10; call++) {
        noise_reduction(in, out, 400);
        long long sum_sq = 0;
        for (int i = 0; i < 400; i++) sum_sq += (int)out[i] * (int)out[i];
        printf("call=%d rms=%.1f dBFS\n", call, 20*log10(sqrt(sum_sq/400.0)/32768.0));
    }
    noise_deinit(); return 0;
}
