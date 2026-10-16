# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Provide double precision math operations for external-field callbacks."""

import warp as wp


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::expm1(x);
#else
return __builtin_expm1(x);
#endif
"""
)
def expm1(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::log1p(x);
#else
return __builtin_log1p(x);
#endif
"""
)
def log1p(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::hypot(x, y);
#else
return __builtin_hypot(x, y);
#endif
"""
)
def hypot(x: wp.float64, y: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::asinh(x);
#else
return __builtin_asinh(x);
#endif
"""
)
def asinh(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::acosh(x);
#else
return __builtin_acosh(x);
#endif
"""
)
def acosh(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::atanh(x);
#else
return __builtin_atanh(x);
#endif
"""
)
def atanh(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::tgamma(x);
#else
return __builtin_tgamma(x);
#endif
"""
)
def gamma(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::lgamma(x);
#else
return __builtin_lgamma(x);
#endif
"""
)
def lgamma(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::fmod(x, y);
#else
return __builtin_fmod(x, y);
#endif
"""
)
def fmod(x: wp.float64, y: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::remainder(x, y);
#else
return __builtin_remainder(x, y);
#endif
"""
)
def remainder(x: wp.float64, y: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return ::ldexp(x, exponent);
#else
return __builtin_ldexp(x, exponent);
#endif
"""
)
def ldexp(x: wp.float64, exponent: int) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
int exponent; return ::frexp(x, &exponent);
#else
int exponent; return __builtin_frexp(x, &exponent);
#endif
"""
)
def _frexp_mantissa(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
int exponent; ::frexp(x, &exponent); return exponent;
#else
int exponent; __builtin_frexp(x, &exponent); return exponent;
#endif
"""
)
def _frexp_exponent(x: wp.float64) -> int:
    pass


@wp.func
def frexp(x: wp.float64):
    return _frexp_mantissa(x), _frexp_exponent(x)


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
double integral; return ::modf(x, &integral);
#else
double integral; return __builtin_modf(x, &integral);
#endif
"""
)
def _modf_fractional(x: wp.float64) -> wp.float64:
    pass


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
double integral; ::modf(x, &integral); return integral;
#else
double integral; __builtin_modf(x, &integral); return integral;
#endif
"""
)
def _modf_integral(x: wp.float64) -> wp.float64:
    pass


@wp.func
def modf(x: wp.float64):
    return _modf_fractional(x), _modf_integral(x)


@wp.func_native(
    "union { unsigned long long bits; double value; } cast; cast.bits = bits; return cast.value;"
)
def float64_from_bits(bits: wp.uint64) -> wp.float64:
    pass


@wp.func_native(
    "union { unsigned int bits; float value; } cast; cast.bits = (unsigned int)bits; return cast.value;"
)
def float32_from_bits(bits: wp.uint64) -> wp.float32:
    pass


@wp.func
def _float_divmod(a: wp.float64, b: wp.float64):
    remainder_value = fmod(a, b)
    quotient = (a - remainder_value) / b
    if remainder_value != wp.float64(0.0):
        if (b < wp.float64(0.0)) != (remainder_value < wp.float64(0.0)):
            remainder_value += b
            quotient -= wp.float64(1.0)
    else:
        remainder_value = wp.copysign(wp.float64(0.0), b)
    if quotient != wp.float64(0.0):
        floor_value = wp.floor(quotient)
        if quotient - floor_value > wp.float64(0.5):
            floor_value += wp.float64(1.0)
    else:
        floor_value = wp.copysign(wp.float64(0.0), a / b)
    return floor_value, remainder_value


@wp.func
def floor_divide_float64(a: wp.float64, b: wp.float64):
    quotient, remainder_value = _float_divmod(a, b)
    return quotient


@wp.func
def modulo_float64(a: wp.float64, b: wp.float64):
    quotient, remainder_value = _float_divmod(a, b)
    return remainder_value


@wp.func
def floor_divide_int64(a: wp.int64, b: wp.int64):
    # Warp 1.18 floors integer `//` but truncates `%`; other versions may
    # truncate both. Derive the remainder from the quotient, so the floor
    # correction applies only when `//` truncated.
    quotient = a // b
    remainder_value = a - quotient * b
    if remainder_value != wp.int64(0) and (remainder_value < wp.int64(0)) != (
        b < wp.int64(0)
    ):
        quotient -= wp.int64(1)
    return quotient


@wp.func
def modulo_int64(a: wp.int64, b: wp.int64):
    remainder_value = a % b
    if remainder_value != wp.int64(0) and (remainder_value < wp.int64(0)) != (
        b < wp.int64(0)
    ):
        remainder_value += b
    return remainder_value


@wp.func
def integer_power_int64(base: wp.int64, exponent: wp.int64):
    if exponent < wp.int64(0):
        if base == wp.int64(1):
            return wp.int64(1)
        if base == wp.int64(-1):
            return (
                wp.int64(1) if exponent % wp.int64(2) == wp.int64(0) else wp.int64(-1)
            )
        return wp.int64(0)
    result = wp.int64(1)
    factor = base
    power = exponent
    while power > wp.int64(0):
        if power % wp.int64(2) != wp.int64(0):
            result *= factor
        power = power // wp.int64(2)
        if power > wp.int64(0):
            factor *= factor
    return result


@wp.func
def integer_power_uint64(base: wp.uint64, exponent: wp.uint64):
    result = wp.uint64(1)
    factor = base
    power = exponent
    while power > wp.uint64(0):
        if power % wp.uint64(2) != wp.uint64(0):
            result *= factor
        power = power // wp.uint64(2)
        if power > wp.uint64(0):
            factor *= factor
    return result
