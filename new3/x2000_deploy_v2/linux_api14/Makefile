# linux_api13 Makefile -- UL-UNAS X2000 deployment (16kHz internal)
# ================================================================
# Build: make
# Deploy: make deploy  (X2000 at 192.168.42.159)
# PC test: make pc_test

BIN = ulunas_q15

CC = /home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
CFLAGS = -O3 -std=c99 -march=mips32r2 -msoft-float
LDFLAGS = -lm -lrt -static

PC_CC = gcc
PC_CFLAGS = -O3 -std=c99 -DPC_PLATFORM
PC_LDFLAGS = -lm

SRCS = agc.c ulunas_linux.c ulunas_fp.c ulunas_lut.c \
       ulunas_modules.c ulunas_matlab_weights.c ulunas_infer.c ulunas_nr.c

all: $(BIN)

$(BIN): $(SRCS) *.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

pc_test: $(SRCS) *.h
	$(PC_CC) $(PC_CFLAGS) -o pc_test_nr $(SRCS) $(PC_LDFLAGS)
	@echo "PC binary: ./pc_test_nr"
	@echo "Usage: ./pc_test_nr test < input.wav > output.wav"

deploy: $(BIN)
	ssh root@192.168.42.159 "killall $(BIN) 2>/dev/null; sleep 1"
	scp $(BIN) root@192.168.42.159:/data/$(BIN)
	ssh root@192.168.42.159 "cp /data/$(BIN) /data/tianhai/$(BIN) && chmod +x /data/$(BIN) /data/tianhai/$(BIN); nohup /data/$(BIN) > /dev/null 2>&1 &"
	@sleep 2
	@ssh root@192.168.42.159 "ps | grep $(BIN) | grep -v grep"

clean:
	rm -f $(BIN) pc_test_nr