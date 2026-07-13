/* test_wav.c — PC wav denoise test for linux_api10
 * Build: gcc -O2 -std=c99 -I. -DPC_PLATFORM -o test_wav test_wav.c
 *        ulunas_fp.c ulunas_lut.c ulunas_modules.c ulunas_matlab_weights.c ulunas_nr.c -lm
 * Usage: ./test_wav <input.wav> <output.wav> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "noise_reduction.h"

/* Minimal WAV reader/writer for 16-bit mono PCM */
typedef struct { int sr, ch, n; short *d; } wav_t;

static wav_t wav_read(const char *path) {
    wav_t w={0}; FILE *f=fopen(path,"rb"); if(!f)return w;
    uint8_t h[44]; fread(h,1,44,f);
    w.ch=h[22]|(h[23]<<8); w.sr=h[24]|(h[25]<<8)|(h[26]<<16)|(h[27]<<24);
    int ds=w.ch*2, sz=h[40]|(h[41]<<8)|(h[42]<<16)|(h[43]<<24);
    w.n=sz/ds; w.d=malloc(w.n*2);
    fread(w.d,2,w.n,f); fclose(f);
    printf("Read %s: %dHz %dch %d samples\n",path,w.sr,w.ch,w.n);
    return w;
}
static void wav_write(const char *path, wav_t w) {
    FILE *f=fopen(path,"wb");
    int ds=w.ch*2, sz=w.n*ds, bps=16;
    uint8_t h[44]={0x52,0x49,0x46,0x46,0,0,0,0,0x57,0x41,0x56,0x45,
        0x66,0x6d,0x74,0x20,16,0,0,0,1,0,(uint8_t)w.ch,0,
        (uint8_t)(w.sr),(uint8_t)(w.sr>>8),(uint8_t)(w.sr>>16),(uint8_t)(w.sr>>24),
        0,0,0,0,(uint8_t)ds,(uint8_t)(ds>>8),bps,0,0x64,0x61,0x74,0x61,0,0,0,0};
    int bs=w.sr*ds, fs=sz+36;
    memcpy(h+4,&fs,4); memcpy(h+28,&bs,4); memcpy(h+40,&sz,4);
    fwrite(h,1,44,f); fwrite(w.d,2,w.n,f); fclose(f);
    printf("Wrote %s: %d samples\n",path,w.n);
}

int main(int argc, char **argv) {
    if(argc<3){fprintf(stderr,"Usage: %s <in.wav> <out.wav>\n",argv[0]);return 1;}
    wav_t w=wav_read(argv[1]); if(!w.d)return 1;
    noise_init();
    short *out=malloc(w.n*2);
    /* Process in chunks of 400 (50ms @ 8kHz) */
    int chunk=400;
    for(int i=0;i<w.n;i+=chunk){
        int n=(i+chunk>w.n)?w.n-i:chunk;
        noise_reduction(w.d+i, out+i, n);
    }
    noise_deinit();
    wav_t ow={w.sr,w.ch,w.n,out};
    wav_write(argv[2],ow);
    free(w.d);free(out);
    return 0;
}
