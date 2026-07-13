# Makefile for new2/ulunas_c — UL-UNAS Module & Full Pipeline Tests
# =================================================================
# Usage:
#   make all       — Build all tests
#   make test      — Build individual module tests
#   make golden    — Build full pipeline golden verification
#   make run-all   — Build and run all tests
#   make clean
#
# Prerequisites:
#   - MATLAB golden data in dump_matlab/ (from export_all_layers_v2)
#   - Mature project at ../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/

CC       = gcc
CFLAGS   = -O2 -std=c99 -Wall
LDFLAGS  = -lm

# Path to mature project
MATURE_DIR = ../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2

# Include paths
INCLUDES = -I. -I$(MATURE_DIR)

# Mature project sources + weight object
MATURE_SRCS = $(MATURE_DIR)/ulunas_fp.c $(MATURE_DIR)/ulunas_modules.c $(MATURE_DIR)/ulunas_lut.c
WEIGHT_OBJ  = $(MATURE_DIR)/ulunas_matlab_weights.o

# Individual module tests (verified)
MODULE_TESTS = log_gen_test bm_test sigmoid_bs_mask_test istft_test

# Full pipeline golden verification
GOLDEN_TEST  = test_matlab_golden

.PHONY: all test golden run-all clean

all: test golden

test: $(MODULE_TESTS)

golden: $(GOLDEN_TEST)

# Module tests — self-contained (each implements its own operators)
log_gen_test: log_gen_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bm_test: bm_test.c erb_weights.h erb_bm_weight.inc
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sigmoid_bs_mask_test: sigmoid_bs_mask_test.c erb_weights.h erb_bs_weight.inc
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

istft_test: istft_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Full pipeline golden verification — links against mature project operators
$(GOLDEN_TEST): test_matlab_golden.c $(WEIGHT_OBJ) $(MATURE_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(MATURE_SRCS) $(WEIGHT_OBJ) $(LDFLAGS)

# Ensure weight object is up to date
$(WEIGHT_OBJ):
	cd $(MATURE_DIR) && $(MAKE) -f Makefile weight_obj 2>/dev/null || \
	$(CC) -c -O2 -std=c99 $(INCLUDES) -o $@ $(MATURE_DIR)/ulunas_matlab_weights.c

run-all: all
	@echo "===== log_gen_test =====" && ./log_gen_test dump_matlab
	@echo "===== bm_test =====" && ./bm_test dump_matlab
	@echo "===== sigmoid_bs_mask_test =====" && ./sigmoid_bs_mask_test dump_matlab
	@echo "===== test_matlab_golden =====" && ./test_matlab_golden dump_matlab

clean:
	rm -f $(MODULE_TESTS) $(GOLDEN_TEST) test_matlab_golden
