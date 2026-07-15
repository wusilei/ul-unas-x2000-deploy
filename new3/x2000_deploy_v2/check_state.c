/* check_state.c — Verify ulunas_state_t layout vs what RNN code accesses */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include "ulunas_fp.h"

int main() {
    ulunas_state_t st;
    memset(&st, 0, sizeof(st));

    printf("=== Struct layout ===\n");
    printf("sizeof(ulunas_state_t) = %zu bytes\n", sizeof(st));
    printf("sizeof(int16_t) = %zu, sizeof(int32_t) = %zu\n", sizeof(int16_t), sizeof(int32_t));

    /* Check tfa_cache types */
    printf("\n=== tfa_cache types ===\n");
    printf("tfa_cache_e0 type size: %zu (expected int16_t=2 or int32_t=4)\n", sizeof(st.tfa_cache_e0[0]));
    printf("tfa_cache_e1 type size: %zu\n", sizeof(st.tfa_cache_e1[0]));
    printf("inter_cache_0 type size: %zu\n", sizeof(st.inter_cache_0[0]));

    /* Check offsets */
    printf("\n=== Offsets ===\n");
    printf("inter_cache_0 offset: %zu\n", offsetof(ulunas_state_t, inter_cache_0));
    printf("inter_cache_1 offset: %zu\n", offsetof(ulunas_state_t, inter_cache_1));
    printf("Expected inter_cache_0 end: %zu\n", offsetof(ulunas_state_t, inter_cache_0) + sizeof(st.inter_cache_0));
    printf("inter_cache_1 start: %zu\n", offsetof(ulunas_state_t, inter_cache_1));

    /* Check if offsets are contiguous */
    size_t expected_ic1 = offsetof(ulunas_state_t, inter_cache_0) + sizeof(st.inter_cache_0);
    size_t actual_ic1 = offsetof(ulunas_state_t, inter_cache_1);
    printf("\n=== Contiguity check ===\n");
    printf("inter_cache_0 end = %zu, inter_cache_1 start = %zu, match=%s\n",
           expected_ic1, actual_ic1,
           expected_ic1 == actual_ic1 ? "YES" : "NO — GAP OR OVERLAP!");

    /* Check RNN access: inter_rnn_module reads inter_cache_0 as [33][16] */
    printf("\n=== RNN access pattern ===\n");
    int16_t *ic = st.inter_cache_0;
    printf("inter_cache_0 total elements: %zu\n", sizeof(st.inter_cache_0)/sizeof(st.inter_cache_0[0]));
    printf("Expected: 33*16 = 528\n");
    printf("Actual: %zu\n", sizeof(st.inter_cache_0)/sizeof(st.inter_cache_0[0]));

    /* Check the full struct size matches expectation */
    size_t expected_size = 0;
    expected_size += sizeof(int32_t) * (2*129 + 24*65 + 24*33);  /* conv_cache_e0/1/2 */
    expected_size += sizeof(int32_t) * (24*33 + 12*33 + 12*2*65); /* conv_cache_d0/1/2 */
    expected_size += sizeof(st.tfa_cache_e0[0]) * (24+48+48+64+32); /* tfa e0-e4 */
    expected_size += sizeof(st.tfa_cache_e0[0]) * (64+48+48+24+2);  /* tfa d0-d4 */
    expected_size += sizeof(st.inter_cache_0[0]) * (33*16 + 33*16); /* inter 0/1 */
    printf("\nExpected struct size: %zu, Actual: %zu, Match: %s\n",
           expected_size, sizeof(st),
           expected_size == sizeof(st) ? "YES" : "NO");

    /* Test: write known pattern, read back via RNN's expected layout */
    printf("\n=== Access pattern test ===\n");
    /* Mark inter_cache_0 with element index */
    for (int i = 0; i < 33*16; i++) st.inter_cache_0[i] = (int16_t)i;
    /* Read as if RNN: inter_rnn_module accesses inter_cache as [33][16] */
    /* GRU hh path: h_cache[i] where i is nHidden index */
    int errors = 0;
    for (int t = 0; t < 33; t++) {       /* time steps */
        for (int c = 0; c < 16; c++) {   /* channels */
            int flat_idx = t * 16 + c;     /* row-major [T][C] */
            int expected = t * 16 + c;
            if ((int)st.inter_cache_0[flat_idx] != expected) {
                if (errors < 3) printf("  MISMATCH: inter_cache[%d][%d] = %d, expected %d\n",
                    t, c, st.inter_cache_0[flat_idx], expected);
                errors++;
            }
        }
    }
    printf("Row-major [33][16] access: %d errors\n", errors);

    /* Check col-major access */
    errors = 0;
    for (int t = 0; t < 33; t++) {
        for (int c = 0; c < 16; c++) {
            int col_idx = c + 16 * t;   /* col-major: c changes fastest */
            /* Value was stored row-major, so col_idx != t*16+c */
        }
    }

    /* Write known Q20 values (larger than int16_t) to test truncation */
    printf("\n=== int16_t truncation test ===\n");
    int32_t q20_val = 250000;  /* Q20 value > 32767, would be truncated by int16_t */
    int16_t *ic_ptr = (int16_t*)&st.inter_cache_0[0];
    *((int32_t*)ic_ptr) = q20_val;  /* Write Q20 value through int32_t pointer */
    printf("Wrote %d to inter_cache_0[0..1], read back as int16_t: [0]=%d, [1]=%d\n",
           q20_val, st.inter_cache_0[0], st.inter_cache_0[1]);
    printf("If [0] != %d or [1] != 0, int32_t->int16_t access mismatch exists.\n",
           (int16_t)q20_val);

    return 0;
}
