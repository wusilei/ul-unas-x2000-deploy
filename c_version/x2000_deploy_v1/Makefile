# Makefile — UL-UNAS C Fixed-Point Implementation
# ===============================================
# PC verification build (gcc) and X2000 cross-compile

CC_PC = gcc
CC_X2000 = mips-linux-gnu-gcc
CFLAGS = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm

# Source files
SRCS = ulunas_fp.c ulunas_modules.c ulunas_infer.c ulunas_matlab_weights.c

# Targets
.PHONY: all clean test pc x2000

all: pc

# PC build
pc: test_matlab_golden

test_matlab_golden: test_matlab_golden.c $(SRCS) ulunas_fp.h ulunas_matlab_weights.h
	$(CC_PC) $(CFLAGS) -o $@ test_matlab_golden.c $(SRCS) $(LDFLAGS)

# Verify compilation only (no weights)
check-build: $(SRCS) ulunas_fp.h
	$(CC_PC) $(CFLAGS) -fsyntax-only ulunas_fp.c $(LDFLAGS)
	@echo "Syntax check passed."

# X2000 cross-compile
x2000: ulunas_x2000

ulunas_x2000: $(SRCS) ulunas_fp.h ulunas_matlab_weights.h
	$(CC_X2000) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# Run tests
test: test_matlab_golden
	./test_matlab_golden dump_matlab/

# Clean
clean:
	rm -f test_matlab_golden ulunas_x2000 *.o
