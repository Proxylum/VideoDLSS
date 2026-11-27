// TASK-0080 (HDR step TC): the AVX2 row of the HDR -> SDR transfer - 8 pixels at a time, the same bits as the portable row
// (HdrTransfer::Apply after ColorConvert's Y'CbCr -> R'G'B'): every operation of the portable row in its order, IEEE division and square
// root, no FMA, gathers only to read tables. Built with /arch:AVX2 (core/CMakeLists.txt) and called only when hdrrow::HasAvx2(); it
// includes nothing but HdrRow.h and the intrinsics, and its helpers have internal linkage, so no inline function built here can stand in
// for a portable one.
//
// The portable row's std::min(a, b) is (b < a) ? b : a = _mm256_min_ps(b, a), std::max(a, b) is (a < b) ? b : a = _mm256_max_ps(b, a)
// (the operand order keeps the sign of a zero), std::clamp(v, lo, hi) is min(max(v, lo), hi), (int)x is _mm256_cvttps_epi32.
#include "convert/HdrRow.h"

#include <immintrin.h>
#include <intrin.h>

namespace dlssvid::hdrrow {

namespace {

using V = __m256;
using Vi = __m256i;

inline V S(float x) { return _mm256_set1_ps(x); }
inline Vi Si(int x) { return _mm256_set1_epi32(x); }
inline V Add(V a, V b) { return _mm256_add_ps(a, b); }
inline V Sub(V a, V b) { return _mm256_sub_ps(a, b); }
inline V Mul(V a, V b) { return _mm256_mul_ps(a, b); }
inline V Div(V a, V b) { return _mm256_div_ps(a, b); }
inline V Min(V a, V b) { return _mm256_min_ps(b, a); }  // std::min(a, b)
inline V Max(V a, V b) { return _mm256_max_ps(b, a); }  // std::max(a, b)
inline V Clamp(V v, V lo, V hi) { return Min(Max(v, lo), hi); }
inline V Ge(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
inline V Gt(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
inline V Lt(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
inline V Le(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
inline V Eq(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
inline V Sel(V mask, V yes, V no) { return _mm256_blendv_ps(no, yes, mask); }  // mask ? yes : no
inline Vi Trunc(V x) { return _mm256_cvttps_epi32(x); }
inline V ToF(Vi i) { return _mm256_cvtepi32_ps(i); }
inline Vi ClampI(Vi i, int lo, int hi) { return _mm256_max_epi32(_mm256_min_epi32(i, Si(hi)), Si(lo)); }  // std::max(std::min(i, hi), lo)
inline V G(const float* t, Vi i) { return _mm256_i32gather_ps(t, i, 4); }
inline Vi Gi(const int32_t* t, Vi i) { return _mm256_i32gather_epi32(reinterpret_cast<const int*>(t), i, 4); }
inline Vi Count(V mask, Vi n) { return _mm256_sub_epi32(n, _mm256_castps_si256(mask)); }  // n + (mask ? 1 : 0)
inline Vi Wrap1(Vi i, int n) {  // i + 1 == n ? 0 : i + 1
    const Vi i1 = _mm256_add_epi32(i, Si(1));
    return _mm256_andnot_si256(_mm256_cmpeq_epi32(i1, Si(n)), i1);
}

// hdrrow_detail::Lerp
inline V Lerp(const float* t, int n, V x) {
    const V fx = Mul(x, S(static_cast<float>(n)));
    const Vi i = ClampI(Trunc(fx), 0, n - 1);
    const V a = G(t, i), b = G(t + 1, i);
    return Add(a, Mul(Sub(b, a), Sub(fx, ToF(i))));
}

// hdrrow_detail::OctaveLerp
inline V OctaveLerp(const float* t, V x, float zero) {
    const float kLow = 2.3283064365386963e-10f;
    const V low = Add(S(zero), Mul(S(t[0] - zero), Mul(Max(x, _mm256_setzero_ps()), S(4294967296.f))));
    const V c = Min(Max(x, S(kLow)), S(1.f));
    const Vi b = _mm256_castps_si256(c);
    const Vi idx = _mm256_add_epi32(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_srli_epi32(b, 23), Si(95)), 8),
                                    _mm256_and_si256(_mm256_srli_epi32(b, 15), Si(255)));
    const Vi i = _mm256_min_epi32(idx, Si(32 * 256));
    const V a = G(t, i), a1 = G(t + 1, i);
    const V frac = Mul(ToF(_mm256_and_si256(b, Si(0x7FFF))), S(1.f / 32768.f));
    return Sel(Lt(x, S(kLow)), low, Add(a, Mul(Sub(a1, a), frac)));
}

// hdrrow_detail::PqDec
inline V PqDec(const Step& g, V e) {
    const V x = Mul(Clamp(e, _mm256_setzero_ps(), S(g.iw)), S(g.decScale));
    const Vi i = ClampI(Trunc(x), 0, 4095);
    const V a = G(g.dec, i), b = G(g.dec + 1, i);
    return Add(a, Mul(Sub(b, a), Sub(x, ToF(i))));
}

inline V Neg(V x) { return _mm256_xor_ps(x, S(-0.f)); }

// hdrrow_detail::PseudoHue: base + num / den of the quadrant
inline V PseudoHue(V ct, V cp) {
    const V zero = _mm256_setzero_ps();
    const V cpGe = Ge(cp, zero), ctGe = Ge(ct, zero), ctLt = Lt(ct, zero);
    const V baseU = Sel(ctGe, S(0.f), S(1.f)), numU = Sel(ctGe, cp, Neg(ct)), denU = Sel(ctGe, Add(ct, cp), Sub(cp, ct));
    const V baseL = Sel(ctLt, S(2.f), S(3.f)), numL = Sel(ctLt, Neg(cp), ct), denL = Sel(ctLt, Sub(Neg(ct), cp), Sub(ct, cp));
    return Add(Sel(cpGe, baseU, baseL), Div(Sel(cpGe, numU, numL), Sel(cpGe, denU, denL)));
}

inline V RelativePa(const Axis& a, V pa) {
    const V q = Sub(pa, S(a.pa0));
    return Add(q, _mm256_and_ps(Lt(q, _mm256_setzero_ps()), S(4.f)));
}

struct AxisAt {
    Vi g0, g1;
    V tg, u;
};

// hdrrow_detail::AxisLookup
inline AxisAt AxisLookup(const Step& g, const Axis& a, V I, V q) {
    Vi s = _mm256_setzero_si256();
    for (int i = 1; i < 6; ++i) s = Count(Ge(q, S(a.vertex[i])), s);
    const V fg = Add(ToF(Gi(a.start, s)), Mul(Sub(q, G(a.vertex, s)), G(a.scale, s)));
    AxisAt r;
    r.g0 = ClampI(Trunc(fg), 0, a.n - 1);
    r.g1 = Wrap1(r.g0, a.n);
    r.tg = Sub(fg, ToF(r.g0));
    const V c0 = G(a.cuspI, r.g0), c1 = G(a.cuspI, r.g1);
    const V ic = Add(c0, Mul(Sub(c1, c0), r.tg));
    const V below = Mul(Div(Sub(I, S(g.e0)), Sub(ic, S(g.e0))), S(g.uc));
    const V above = Add(S(g.uc), Mul(Div(Sub(I, ic), Sub(S(g.iw), ic)), S(g.ucRest)));
    r.u = Sel(Le(I, ic), below, above);
    return r;
}

// hdrrow_detail::B709At
inline V B709At(const Step& g, V I, V pa) {
    const Bounds709& b = g.b709;
    const V q = RelativePa(b.axis, pa);
    const AxisAt ax = AxisLookup(g, b.axis, I, q);
    const V fi = Mul(Clamp(ax.u, _mm256_setzero_ps(), S(1.f)), S(127.f));
    const Vi i0 = ClampI(Trunc(fi), 0, 126);
    const V ti = Sub(fi, ToF(i0));
    alignas(32) float k[13][8];
    const Vi base = _mm256_mullo_epi32(i0, Si(12));
    for (int m = 0; m < 12; ++m) {
        const V a = G(b.knot, _mm256_add_epi32(base, Si(m))), c = G(b.knot, _mm256_add_epi32(base, Si(m + 12)));
        _mm256_store_ps(k[m], Add(a, Mul(Sub(c, a), ti)));
    }
    const V k0 = _mm256_load_ps(k[0]);
    _mm256_store_ps(k[12], Add(k0, S(4.f)));
    V dq = Sub(q, k0);
    dq = Add(dq, _mm256_and_ps(Lt(dq, _mm256_setzero_ps()), S(4.f)));
    dq = Sub(dq, _mm256_and_ps(Ge(dq, S(4.f)), S(4.f)));
    const V qq = Add(k0, dq);
    Vi c = _mm256_setzero_si256();
    for (int m = 1; m < 12; ++m) c = Count(Ge(qq, _mm256_load_ps(k[m])), c);
    const Vi lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const Vi at = _mm256_add_epi32(_mm256_slli_epi32(c, 3), lane);
    const V kc = G(&k[0][0], at), kc1 = G(&k[0][0], _mm256_add_epi32(at, Si(8)));
    const V fp = Add(ToF(Gi(b.start, c)), Div(Mul(Sub(qq, kc), G(b.cols, c)), Sub(kc1, kc)));
    const Vi j0 = ClampI(Trunc(fp), 0, b.n - 1);
    const Vi j1 = Wrap1(j0, b.n);
    const V tp = Sub(fp, ToF(j0));
    const Vi r0 = _mm256_mullo_epi32(i0, Si(b.n)), r1 = _mm256_add_epi32(r0, Si(b.n));
    const V n00 = G(b.node, _mm256_add_epi32(r0, j0)), n01 = G(b.node, _mm256_add_epi32(r0, j1));
    const V n10 = G(b.node, _mm256_add_epi32(r1, j0)), n11 = G(b.node, _mm256_add_epi32(r1, j1));
    const V lo = Add(n00, Mul(Sub(n01, n00), tp)), hi = Add(n10, Mul(Sub(n11, n10), tp));
    return Add(lo, Mul(Sub(hi, lo), ti));
}

// hdrrow_detail::B2020At
inline V B2020At(const Step& g, V I, V pa) {
    const Bounds2020& b = g.b2020;
    const V q = RelativePa(b.axis, pa);
    const AxisAt ax = AxisLookup(g, b.axis, I, q);
    const V fi = Mul(Clamp(ax.u, _mm256_setzero_ps(), S(1.f)), S(127.f));
    const Vi i0 = ClampI(Trunc(fi), 0, 126);
    const V ti = Sub(fi, ToF(i0));
    const Vi r0 = _mm256_mullo_epi32(i0, Si(b.axis.n)), r1 = _mm256_add_epi32(r0, Si(b.axis.n));
    const V n00 = G(b.node, _mm256_add_epi32(r0, ax.g0)), n01 = G(b.node, _mm256_add_epi32(r0, ax.g1));
    const V n10 = G(b.node, _mm256_add_epi32(r1, ax.g0)), n11 = G(b.node, _mm256_add_epi32(r1, ax.g1));
    const V lo = Add(n00, Mul(Sub(n01, n00), ax.tg)), hi = Add(n10, Mul(Sub(n11, n10), ax.tg));
    return Add(lo, Mul(Sub(hi, lo), ti));
}

inline V Row3(const float* m, int row, V x0, V x1, V x2) {  // (m0 x0 + m1 x1) + m2 x2
    return Add(Add(Mul(S(m[3 * row]), x0), Mul(S(m[3 * row + 1]), x1)), Mul(S(m[3 * row + 2]), x2));
}

// HdrGamut::Map, first part: a grey is itself, a level-1 colour is T l; `done` marks both
struct Prefix {
    V t[3];
    V grey, done;
};

inline Prefix MapPrefix(const Step& g, const V l[3]) {
    const V zero = _mm256_setzero_ps(), one = S(1.f);
    Prefix p;
    p.grey = _mm256_and_ps(Eq(l[0], l[1]), Eq(l[1], l[2]));
    for (int c = 0; c < 3; ++c) p.t[c] = Row3(g.t, c, l[0], l[1], l[2]);
    // HdrGamut::Level1
    const V Y = Add(Add(Mul(S(g.w709[0]), p.t[0]), Mul(S(g.w709[1]), p.t[1])), Mul(S(g.w709[2]), p.t[2]));
    const V mn = Min(p.t[0], Min(p.t[1], p.t[2])), mx = Max(p.t[0], Max(p.t[1], p.t[2]));
    const V tau = S(g.tau);
    const V level1 = _mm256_and_ps(_mm256_and_ps(Ge(mn, zero), Le(mx, one)),
                                   _mm256_and_ps(Le(Sub(Y, mn), Mul(tau, Y)), Le(Sub(mx, Y), Mul(tau, Sub(one, Y)))));
    p.done = _mm256_or_ps(p.grey, level1);
    return p;
}

// HdrGamut::Map, the rest, for colours that are neither grey nor level 1: inside the knee clamp(T l), above it the compressed colour
inline void MapRest(const Step& g, const V l[3], const V t[3], V out[3]) {
    const V zero = _mm256_setzero_ps(), one = S(1.f);
    const V L = OctaveLerp(g.enc, Row3(g.mlms, 0, l[0], l[1], l[2]), g.e0);
    const V M = OctaveLerp(g.enc, Row3(g.mlms, 1, l[0], l[1], l[2]), g.e0);
    const V Sx = OctaveLerp(g.enc, Row3(g.mlms, 2, l[0], l[1], l[2]), g.e0);
    const V I = Mul(S(0.5f), Add(L, M));
    const V ct = Row3(g.ict, 0, L, M, Sx), cp = Row3(g.ict, 1, L, M, Sx);
    const V C = _mm256_sqrt_ps(Add(Mul(ct, ct), Mul(cp, cp)));
    const V pa = Sel(Gt(C, zero), PseudoHue(ct, cp), zero);
    const V b7 = B709At(g, I, pa), kb = Mul(S(g.knee), b7);
    const V inside = _mm256_or_ps(_mm256_xor_ps(Gt(C, kb), _mm256_castsi256_ps(Si(-1))),
                                  _mm256_xor_ps(Gt(b7, zero), _mm256_castsi256_ps(Si(-1))));
    const V b20 = Max(B2020At(g, I, pa), b7);
    const V a = Mul(S(g.kneeRest), b7), D = Max(Sub(b20, kb), a), dd = Min(Sub(C, kb), D);
    const V q = Div(Sub(Mul(D, D), Mul(a, a)), Mul(Mul(Mul(a, a), D), D));
    const V f = Div(Add(kb, Div(dd, _mm256_sqrt_ps(Add(one, Mul(Mul(dd, dd), q))))), C);
    const V fct = Mul(f, ct), fcp = Mul(f, cp);
    const V Lo = PqDec(g, Add(Add(I, Mul(S(g.ictInv[0]), fct)), Mul(S(g.ictInv[1]), fcp)));
    const V Mo = PqDec(g, Add(Add(I, Mul(S(g.ictInv[2]), fct)), Mul(S(g.ictInv[3]), fcp)));
    const V So = PqDec(g, Add(Add(I, Mul(S(g.ictInv[4]), fct)), Mul(S(g.ictInv[5]), fcp)));
    for (int c = 0; c < 3; ++c) out[c] = Sel(inside, Clamp(t[c], zero, one), Clamp(Row3(g.a709, c, Lo, Mo, So), zero, one));
}

constexpr int kPoints = 4096;

// HdrTransfer::Linear on codes in [0, 1]
inline void Linear(const Chain& ch, V r, V gc, V b, V l[3]) {
    V L[3], Gain;
    if (!ch.hlg) {
        L[0] = Lerp(ch.eotf, kPoints, r), L[1] = Lerp(ch.eotf, kPoints, gc), L[2] = Lerp(ch.eotf, kPoints, b);
        Gain = Lerp(ch.gain, kPoints, Max(r, Max(gc, b)));
    } else {
        const V E0 = Lerp(ch.eotf, kPoints, r), E1 = Lerp(ch.eotf, kPoints, gc), E2 = Lerp(ch.eotf, kPoints, b);
        const V ys = Add(Add(Mul(S(ch.w[0]), E0), Mul(S(ch.w[1]), E1)), Mul(S(ch.w[2]), E2));
        const V k = Mul(S(ch.hlgScale), OctaveLerp(ch.ys02, ys, 0.f));
        L[0] = Mul(k, E0), L[1] = Mul(k, E1), L[2] = Mul(k, E2);
        Gain = Lerp(ch.gain, kPoints, Min(Mul(Max(L[0], Max(L[1], L[2])), S(ch.gainScale)), S(1.f)));
    }
    for (int c = 0; c < 3; ++c) l[c] = Min(Mul(L[c], Gain), S(1.f));
}

// the last step of HdrTransfer::Apply
inline V Oetf(const Chain& ch, V l) { return Lerp(ch.oetf, kPoints, _mm256_sqrt_ps(Clamp(l, _mm256_setzero_ps(), S(1.f)))); }

inline V Load8(const uint8_t* p) { return ToF(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)))); }
inline V Load8(const uint16_t* p) { return ToF(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))); }
inline V LoadHalf(const uint8_t* p) {  // 4 samples, each twice
    const Vi v = _mm256_cvtepu8_epi32(_mm_loadu_si32(p));
    return ToF(_mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)));
}
inline V LoadHalf(const uint16_t* p) {
    const Vi v = _mm256_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
    return ToF(_mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)));
}

// Two passes over the row. The first: Y'CbCr -> R'G'B' -> the chain, and for 8 pixels at a time the step's first part; greys and level-1
// colours are finished there, a vector of 8 slow pixels too, the slow pixels of a mixed vector wait in `todo` (their light in `scratch`).
// The second runs the step's slow part over the waiting pixels 8 at a time - a vector with one slow pixel would otherwise cost 8 slow
// pixels. Every pixel is computed alone, so the order changes no bit.
template <typename T>
uint32_t Row(const Chain& ch, const Yuv& k, const T* y, const T* u, const T* v, int sx, uint32_t width, float* r, float* g, float* b,
             float* scratch, int32_t* todo) {
    if (sx != 0 && sx != 1) return 0;
    const V zero = _mm256_setzero_ps(), one = S(1.f);
    uint32_t x = 0, waiting = 0;
    float* out[3] = {r, g, b};
    for (; x + 8 <= width; x += 8) {
        const V yv = Load8(y + x);
        const V uv = sx ? LoadHalf(u + (x >> 1)) : Load8(u + x);
        const V vv = sx ? LoadHalf(v + (x >> 1)) : Load8(v + x);
        // ColorConvert's ForEachRgb
        const V yy = Mul(Sub(yv, S(k.yOff)), S(k.yScale));
        const V cb = Mul(Sub(uv, S(k.cOff)), S(k.cScale));
        const V cr = Mul(Sub(vv, S(k.cOff)), S(k.cScale));
        const V rc = Clamp(Add(yy, Mul(S(k.crR), cr)), zero, one);
        const V gc = Clamp(Add(Add(yy, Mul(S(k.cbG), cb)), Mul(S(k.crG), cr)), zero, one);
        const V bc = Clamp(Add(yy, Mul(S(k.cbB), cb)), zero, one);
        V l[3];
        Linear(ch, rc, gc, bc, l);
        if (!ch.gamut) {
            for (int c = 0; c < 3; ++c) _mm256_storeu_ps(out[c] + x, Oetf(ch, l[c]));
            continue;
        }
        const Prefix p = MapPrefix(*ch.gamut, l);
        unsigned long pending = static_cast<unsigned long>(~_mm256_movemask_ps(p.done) & 0xFF);
        if (pending == 0xFF) {  // all 8 slow: no queue
            V o[3];
            MapRest(*ch.gamut, l, p.t, o);
            for (int c = 0; c < 3; ++c) _mm256_storeu_ps(out[c] + x, Oetf(ch, o[c]));
            continue;
        }
        for (int c = 0; c < 3; ++c) _mm256_storeu_ps(out[c] + x, Oetf(ch, Sel(p.grey, l[c], p.t[c])));  // the waiting ones: overwritten
        if (pending) {
            for (int c = 0; c < 3; ++c) _mm256_storeu_ps(scratch + static_cast<size_t>(c) * width + x, l[c]);
            while (pending) {
                unsigned long j;
                _BitScanForward(&j, pending);
                todo[waiting++] = static_cast<int32_t>(x + j);
                pending &= pending - 1;
            }
        }
    }
    for (uint32_t i = 0; i < waiting; i += 8) {
        alignas(32) int32_t at[8];
        for (uint32_t j = 0; j < 8; ++j) at[j] = todo[i + j < waiting ? i + j : waiting - 1];  // a short tail repeats its last pixel
        const Vi vi = _mm256_load_si256(reinterpret_cast<const __m256i*>(at));
        V l[3], t[3], o[3];
        for (int c = 0; c < 3; ++c) l[c] = G(scratch + static_cast<size_t>(c) * width, vi);
        for (int c = 0; c < 3; ++c) t[c] = Row3(ch.gamut->t, c, l[0], l[1], l[2]);
        MapRest(*ch.gamut, l, t, o);
        alignas(32) float res[3][8];
        for (int c = 0; c < 3; ++c) _mm256_store_ps(res[c], Oetf(ch, o[c]));
        for (uint32_t j = 0; j < 8 && i + j < waiting; ++j)
            for (int c = 0; c < 3; ++c) out[c][at[j]] = res[c][j];
    }
    return x;
}

}  // namespace

uint32_t RowAvx2(const Chain& chain, const Yuv& k, const uint8_t* y, const uint8_t* u, const uint8_t* v, int sx, uint32_t width,
                 float* r, float* g, float* b, float* scratch, int32_t* todo) {
    return Row(chain, k, y, u, v, sx, width, r, g, b, scratch, todo);
}

uint32_t RowAvx2(const Chain& chain, const Yuv& k, const uint16_t* y, const uint16_t* u, const uint16_t* v, int sx, uint32_t width,
                 float* r, float* g, float* b, float* scratch, int32_t* todo) {
    return Row(chain, k, y, u, v, sx, width, r, g, b, scratch, todo);
}

}  // namespace dlssvid::hdrrow
