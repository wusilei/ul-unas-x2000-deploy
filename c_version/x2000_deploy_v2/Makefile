# Makefile for UL-UNAS X2000 Fixed-Point Deployment
# ==================================================
# PC build (gcc):   make pc
# X2000 cross:      make x2000 CROSS=mips-linux-gnu-
# Clean:            make clean

CC       = gcc
CROSS   ?=
TARGET   = test_matlab_golden
INFER_T  = ulunas_infer_test

# Common flags
CFLAGS   = -Wall -Wextra -O2 -std=c99
LDFLAGS  = -lm

# Source files
SRCS     = ulunas_fp.c ulunas_lut.c ulunas_modules.c ulunas_matlab_weights.c
OBJS     = $(SRCS:.c=.o)

# MIPS cross-compilation flags
ifeq ($(CROSS),)
  CC_PREFIX =
else
  CC_PREFIX = $(CROSS)
  CFLAGS   += -mips32r2 -mno-float -EL
endif

.PHONY: all pc x2000 clean

all: pc

pc: CFLAGS += -DPC_PLATFORM
pc: $(TARGET)

x2000: CFLAGS += -DX2000_PLATFORM -ffast-math
x2000: $(TARGET)

# Main golden test
$(TARGET): test_matlab_golden.c $(OBJS) ulunas_fp.h ulunas_lut.h
	$(CC_PREFIX)$(CC) $(CFLAGS) -o $@ test_matlab_golden.c $(OBJS) $(LDFLAGS)

# Inference test (single frame)
$(INFER_T): ulunas_infer_test.c ulunas_infer.c $(OBJS) ulunas_fp.h ulunas_lut.h
	$(CC_PREFIX)$(CC) $(CFLAGS) -o $@ ulunas_infer_test.c ulunas_infer.c $(OBJS) $(LDFLAGS)

# Object files
%.o: %.c ulunas_fp.h ulunas_lut.h qr_config.h layer_dims.h ulunas_matlab_weights.h
	$(CC_PREFIX)$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(INFER_T) *.o
