# linux_api Makefile — UL-UNAS X2000 deployment
# ==============================================
# Build: make
# Deploy: make deploy  (X2000 at 192.168.42.159)

BIN = ulunas_q15

CC = /home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
CFLAGS = -O3 -std=c99 -march=mips32r2 -msoft-float
LDFLAGS = -lm -lrt -static

SRCS = agc.c ulunas_linux.c ulunas_fp.c ulunas_lut.c \
       ulunas_modules.c ulunas_matlab_weights.c ulunas_infer.c ulunas_nr.c

all: $(BIN)

$(BIN): $(SRCS) *.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

deploy: $(BIN)
	ssh root@192.168.42.159 "killall $(BIN) 2>/dev/null; sleep 1"
	scp $(BIN) root@192.168.42.159:/data/$(BIN)
	ssh root@192.168.42.159 "chmod +x /data/$(BIN); nohup /data/$(BIN) > /dev/null 2>&1 &"
	@sleep 2
	@ssh root@192.168.42.159 "ps | grep $(BIN) | grep -v grep"

clean:
	rm -f $(BIN)
