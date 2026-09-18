#pragma once

#include <cstdint>
#include <cstring>

namespace dlssvid {

// IEEE 754 binary16 <-> binary32, round to nearest even. Header-only so passes and
// converters can use it without pulling Imath into every translation unit.
inline uint16_t FloatToHalf(float value) {
    uint32_t f;
    std::memcpy(&f, &value, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;

    if (((f >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);  // underflow -> zero
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = static_cast<uint32_t>(exp) << 10 | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // may carry into exponent (correct)
    return static_cast<uint16_t>(sign | half);
}

inline float HalfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {  // subnormal half -> normalized float
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            f = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

}  // namespace dlssvid
