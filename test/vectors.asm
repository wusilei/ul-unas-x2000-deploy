; ================================================================
; vectors.asm — C64x+ Minimal Reset Vector Table
; ================================================================
; 提供复位向量, 跳转到 C 运行时 _c_int00 初始化.
; C64x+: 复位向量位于 L2 起始地址 0x00800000.
; ================================================================

    .global _c_int00
    .global _vectors

    .sect ".vectors"

_vectors:
_vector_reset:
    MVKL _c_int00, B0
    MVKH _c_int00, B0
    B    B0
    NOP  5

_vector_nmi:
    B    _vector_nmi     ; spin on NMI
    NOP  5

_vector_resv2:
    B    _vector_resv2
    NOP  5

_vector_resv3:
    B    _vector_resv3
    NOP  5

_vector_int4:
    B    _vector_int4
    NOP  5

_vector_int5:
    B    _vector_int5
    NOP  5

_vector_int6:
    B    _vector_int6
    NOP  5

_vector_int7:
    B    _vector_int7
    NOP  5

_vector_int8:
    B    _vector_int8
    NOP  5

_vector_int9:
    B    _vector_int9
    NOP  5

_vector_int10:
    B    _vector_int10
    NOP  5

_vector_int11:
    B    _vector_int11
    NOP  5

_vector_int12:
    B    _vector_int12
    NOP  5

_vector_int13:
    B    _vector_int13
    NOP  5

_vector_int14:
    B    _vector_int14
    NOP  5

_vector_int15:
    B    _vector_int15
    NOP  5
