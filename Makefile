# UL-UNAS DM3730 DSP 移植 Makefile
# ===================================
# 编译器: TI CGT C6000 v6.0.8 (CCS3.3 SR12 内置)
# 芯片:   DM3730 C64x+ DSP @ 600MHz, L2 64KB
# 注意:   UL-UNAS 需 DDR spill (L2 64KB 不够)

CGT      = E:/app/CCStudio_v3.3/C6000/cgtools
CC       = $(CGT)/bin/cl6x
AR       = $(CGT)/bin/ar6x

DSPLIB   = E:/app/CCStudio_v3.3/c6400/dsplib
DSPLIB_INC = $(DSPLIB)/include
DSPLIB_LIB = $(DSPLIB)/lib/dsp64x.lib

CFLAGS   = -mv64plus -O3 --opt_for_speed=4 -mw -pm -mt -ml3 --mem_model:data=far -fg -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
ARFLAGS  = r

MODEL    = src/model/ulunas_fp.c src/model/ulunas_modules.c \
           src/model/ulunas_infer.c src/model/ulunas_lut.c \
           src/model/ulunas_matlab_weights.c
PIPELINE = src/pipeline/ulunas_nr.c
SHARED   = src/shared/agc.c
DSP      = src/dsp/dsp_mcbsp.c src/dsp/dsp_link.c src/dsp/dsp_main.c
SRCS     = $(MODEL) $(PIPELINE) $(SHARED) $(DSP)
INCLUDES = -Isrc/model -Isrc/pipeline -Isrc/shared -Isrc/dsp -I$(DSPLIB_INC)

OBJS = ulunas_fp.obj ulunas_modules.obj ulunas_infer.obj \
       ulunas_lut.obj ulunas_matlab_weights.obj ulunas_nr.obj \
       agc.obj dsp_mcbsp.obj dsp_link.obj dsp_main.obj

LIB  = lib/libulunas_dm3730.lib
OUT  = ulunas_dsp.out

all: $(LIB)

$(LIB): $(SRCS)
	@echo "=== Compiling UL-UNAS for DM3730 C64x+ ==="
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRCS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	@echo "Library: $@"

dsp: $(LIB)
	$(CC) $(CFLAGS) $(INCLUDES) -z src/dsp/ulunas_link.cmd -o $(OUT) $(OBJS) $(DSPLIB_LIB)
	@echo "Executable: $(OUT)"

clean:
	rm -f *.obj *.out lib/*.lib
