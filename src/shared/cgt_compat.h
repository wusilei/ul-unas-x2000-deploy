/**
 * cgt_compat.h — TI CGT C6000 v6.0.8 兼容层
 * ==========================================
 * 解决 C89 编译器 vs X2000 C99 代码的兼容性问题.
 *
 * 使用方式: 在所有 <stdint.h> / <math.h> 之前 include 此文件
 *   #include "cgt_compat.h"  // 必须第一行
 *   #include <stdint.h>
 *   #include <math.h>
 *
 * CGT v6.0.8 特征:
 *   - C89 编译器 (无混合声明, 无 for(int i), 无 roundf)
 *   - 有 <stdint.h> (库级), 但 C++ 模式需 __STDC_LIMIT_MACROS
 *   - 无 GCC __attribute__ 扩展
 *   - long = 40-bit, long long = 64-bit
 */

#ifndef CGT_COMPAT_H
#define CGT_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── C++ mode: 启用 stdint.h 中的 INT*_MAX/MIN 宏 ──────────────── */
/* C99 标准脚注: C++ 实现只应在 __STDC_LIMIT_MACROS 已定义时定义这些宏 */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

/* ── roundf / round: C99 函数, C89/CGT 不存在 ─────────────────────── */
/* 提供纯整数实现, 避免软浮点运行时依赖 */
#ifndef roundf
static inline float cgt_roundf(float x) {
    if (x >= 0.0f)
        return (float)(int)(x + 0.5f);
    else
        return (float)(int)(x - 0.5f);
}
#define roundf(x) cgt_roundf(x)
#endif

#ifndef round
static inline double cgt_round(double x) {
    if (x >= 0.0)
        return (double)(long)(x + 0.5);
    else
        return (double)(long)(x - 0.5);
}
#define round(x) cgt_round(x)
#endif

/* ── GCC __attribute__ → 空 ──────────────────────────────────────── */
/* CGT 不支持 GCC 属性扩展, 安全忽略 */
#ifndef __attribute__
#define __attribute__(x)
#endif

/* ── typeof: CGT 支持? ───────────────────────────────────────────── */
/* 如需要可在此定义, 当前代码未使用 */

/* ── CGT 特有: restrict → RESTRICT ──────────────────────────────── */
/* C6000 编译器使用 RESTRICT 而非 restrict */
#ifndef restrict
#define restrict  /* C89 has no restrict */
#endif

/* ── bool: C89 无 bool 类型 ──────────────────────────────────────── */
#ifndef __cplusplus
#ifndef bool
#define bool  int
#define true  1
#define false 0
#endif
#endif

/* ── inline: C89 无 inline ──────────────────────────────────────── */
#ifndef inline
#define inline  /* C89 ignores inline */
#endif

#ifdef __cplusplus
}
#endif

#endif /* CGT_COMPAT_H */
