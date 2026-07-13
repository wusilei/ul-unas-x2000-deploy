#!/bin/bash
# linux_api13 deploy script -- 16kHz internal upsampling
# Run from: D:/haidesi/haidesi/ul-unas-x2000-deploy/.../linux_api13/

set -e

BIN=ulunas_q15
X2000=root@192.168.42.159
CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
CFLAGS="-O3 -std=c99 -march=mips32r2 -msoft-float"
LDFLAGS="-lm -lrt -static"
SRCS="agc.c ulunas_linux.c ulunas_fp.c ulunas_lut.c ulunas_modules.c ulunas_matlab_weights.c ulunas_infer.c ulunas_nr.c"

echo "=== 1) Cross-compile ==="
${CC} ${CFLAGS} -o ${BIN} ${SRCS} ${LDFLAGS}

echo "=== 2) Kill old process, upload ==="
ssh ${X2000} "killall ${BIN} 2>/dev/null; sleep 1"
scp ${BIN} ${X2000}:/data/${BIN}
ssh ${X2000} "cp /data/${BIN} /data/tianhai/${BIN} && chmod +x /data/${BIN} /data/tianhai/${BIN}"

echo "=== 3) Start ==="
ssh ${X2000} "nohup /data/${BIN} > /dev/null 2>&1 &"
sleep 2

echo "=== 4) Verify ==="
ssh ${X2000} "ps | grep ${BIN} | grep -v grep"

echo "=== Deploy complete ==="
