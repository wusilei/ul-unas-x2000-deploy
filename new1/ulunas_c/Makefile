# Makefile for ulunas_nr — UL-UNAS Fixed-Point Noise Reduction
# =================================================================
# Usage:
#   make test      — Build PC test binary (gcc)
#   make x2000     — Cross-compile for X2000 (MIPS)
#   make run       — Build and run verification
#   make clean
#
# Prerequisites:
#   - MATLAB golden data in dump_matlab/
#   - noisy_fileid_1.wav in test_wavs/
#   - Weight headers from mature project (see WEIGHT_DIR below)

CC       = gcc
CFLAGS   = -O2 -std=c99 -Wall -DPC_PLATFORM
LDFLAGS  = -lm

# Path to mature project for weight headers
WEIGHT_DIR = ../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2

# Include paths
INCLUDES = -I. -I$(WEIGHT_DIR)

# For full pipeline: link with mature ulunas_matlab_weights.o
WEIGHT_OBJ = $(WEIGHT_DIR)/ulunas_matlab_weights.o

TARGET   = ulunas_nr_test
SOURCES  = ulunas_nr.c

.PHONY: test run clean x2000

test: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Full pipeline test (needs weight object)
full: $(SOURCES) $(WEIGHT_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) -DFULL_PIPELINE -o ulunas_nr_full $< $(WEIGHT_OBJ) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) ../stft/test_wavs/noisy_fileid_1.wav

x2000:
	mips-linux-gnu-gcc -O2 -std=c99 -DPC_PLATFORM -I. -I$(WEIGHT_DIR) \
		-o ulunas_nr_mips ulunas_nr.c -lm

clean:
	rm -f $(TARGET) ulunas_nr_full ulunas_nr_mips
