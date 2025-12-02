#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>
#include <cstring>
#include "Common.hpp"
#include "FastAriphmetics.hpp"

#include <iostream>

namespace FourierTransform
{
	
template <typename T>
inline void CMul (T& r, T& i, T c, T s) noexcept
{
    T tr = r * c - i * s;
    T ti = r * s + i * c;
    r = tr;
    i = ti;
	return;
}

template <typename T>
inline void apply_twiddle (T& r, T& i, int32_t k, int32_t N) noexcept
{
    // W^k = cos(-2*pi*k/64) + j*sin(-2*pi*k/64)
    // W^k = C - jS
    // User CMul(r, i, c, s) computes (r*c - i*s) + j(r*s + i*c)
    // We want to multiply by (C - jS).
    // So we pass c = C, s = -S.
    
    constexpr T PI = static_cast<T>(3.14159265358979323846);
    T angle = static_cast<T>(-2 * k) * PI / static_cast<T>(N); // Positive angle
    T c, s;
    FastCompute::SinCos(angle, s, c);
    
    CMul(r, i, c, s);
}

template <typename T>
inline void Radix2_Butterfly
(
	const T* in,
	T* out, 
	int32_t stride_in, 
	int32_t stride_out
) noexcept
{
    // Inputs
    const T r0 = in[0];
    const T i0 = in[1];
    const T r1 = in[stride_in];
    const T i1 = in[stride_in + 1];

    // Outputs: (X0 + X1), (X0 - X1)
    out[0] = r0 + r1;
    out[1] = i0 + i1;
    
    out[stride_out]     = r0 - r1;
    out[stride_out + 1] = i0 - i1;
}

template <typename T>
inline void Radix3_Butterfly
(
	const T* in, 
	T* out, 
	int32_t stride_in, 
	int32_t stride_out
) noexcept
{
    constexpr T C1 = static_cast<T>(-0.5);
    constexpr T S1 = static_cast<T>(0.8660254037844386); // sin(2pi/3)

    T r0 = in[0];              T i0 = in[1];
    T r1 = in[stride_in];      T i1 = in[stride_in + 1];
    T r2 = in[2*stride_in];    T i2 = in[2*stride_in + 1];

    // t1 = x1 + x2
    T t1_r = r1 + r2;
    T t1_i = i1 + i2;

    // Out[0] = x0 + t1
    out[0] = r0 + t1_r;
    out[1] = i0 + t1_i;

    // m1 = x0 + t1 * (-0.5)
    T m1_r = r0 + t1_r * C1;
    T m1_i = i0 + t1_i * C1;

    // m2 = (x1 - x2) * sin(60)
    T m2_r = (r1 - r2) * S1;
    T m2_i = (i1 - i2) * S1;

    // Out[1] = m1 - j*m2
    // Out[2] = m1 + j*m2
    const int32_t s1 = stride_out;
    
    out[s1]     = m1_r + m2_i;
    out[s1 + 1] = m1_i - m2_r;

    out[2*s1]     = m1_r - m2_i;
    out[2*s1 + 1] = m1_i + m2_r;
}

template <typename T>
inline void Radix4_Butterfly
(
	const T r0, const T i0,
	const T r1, const T i1,
	const T r2, const T i2,
	const T r3, const T i3,
	T* out,
	const int32_t stride_out = 2
) noexcept
{
    const T t1_r = r0 + r2; const T t1_i = i0 + i2;
    const T t2_r = r0 - r2; const T t2_i = i0 - i2;
    const T t3_r = r1 + r3; const T t3_i = i1 + i3;
    const T t4_r = r1 - r3; const T t4_i = i1 - i3;

    out[0 * stride_out + 0] = t1_r + t3_r;
    out[0 * stride_out + 1] = t1_i + t3_i;
    out[1 * stride_out + 0] = t2_r + t4_i;
    out[1 * stride_out + 1] = t2_i - t4_r;
    out[2 * stride_out + 0] = t1_r - t3_r;
    out[2 * stride_out + 1] = t1_i - t3_i;
    out[3 * stride_out + 0] = t2_r - t4_i;
    out[3 * stride_out + 1] = t2_i + t4_r;
	
	return;
}
	
template <typename T>
inline void Radix4_Butterfly (const T* in, T* out, int32_t stride_in = 2, int32_t stride_out = 2) noexcept
{
	const T r0 = in[0 * stride_in + 0];
	const T i0 = in[0 * stride_in + 1];
	const T r1 = in[1 * stride_in + 0];
	const T i1 = in[1 * stride_in + 1];
	const T r2 = in[2 * stride_in + 0];
	const T i2 = in[2 * stride_in + 1];
	const T r3 = in[3 * stride_in + 0];
	const T i3 = in[3 * stride_in + 1];
	
	Radix4_Butterfly (r0, i0, r1, i1, r2, i2, r3, i3, out, stride_out);
	return;
}

template <typename T>
inline void Radix5_Butterfly
(
	const T* in,
	T* out, 
	int32_t stride_in, 
	int32_t stride_out
) noexcept
{
    // Constants based on 2*pi/5 (72 degrees)
    // c1 = (cos(72) + cos(144)) / 2 = -0.25
    // c2 = (cos(72) - cos(144)) / 2 = 0.559016994
    // s1 = sin(72) = 0.951056516
    // s2 = sin(144) = 0.587785252
    
    constexpr T C1 = static_cast<T>(0.309016994374947); // cos(72)
    constexpr T C2 = static_cast<T>(-0.809016994374947); // cos(144)
    constexpr T S1 = static_cast<T>(0.951056516295154); // sin(72)
    constexpr T S2 = static_cast<T>(0.587785252292473); // sin(144)

    // Load inputs
    T r0 = in[0];             T i0 = in[1];
    T r1 = in[stride_in];     T i1 = in[stride_in + 1];
    T r2 = in[2*stride_in];   T i2 = in[2*stride_in + 1];
    T r3 = in[3*stride_in];   T i3 = in[3*stride_in + 1];
    T r4 = in[4*stride_in];   T i4 = in[4*stride_in + 1];

    // Sums and Diffs
    T t1_r = r1 + r4; T t1_i = i1 + i4;
    T t2_r = r1 - r4; T t2_i = i1 - i4;
    T t3_r = r2 + r3; T t3_i = i2 + i3;
    T t4_r = r2 - r3; T t4_i = i2 - i3;

    // Output 0 (DC)
    out[0] = r0 + t1_r + t3_r;
    out[1] = i0 + t1_i + t3_i;

    // Computation
    // M1 = r0 + t1*C1 + t3*C2
    T m1_r = r0 + t1_r * C1 + t3_r * C2;
    T m1_i = i0 + t1_i * C1 + t3_i * C2;

    // M2 = r0 + t1*C2 + t3*C1
    T m2_r = r0 + t1_r * C2 + t3_r * C1;
    T m2_i = i0 + t1_i * C2 + t3_i * C1;

    // M3 = t2*S1 + t4*S2
    T m3_r = t2_r * S1 + t4_r * S2;
    T m3_i = t2_i * S1 + t4_i * S2;

    // M4 = t2*S2 - t4*S1
    T m4_r = t2_r * S2 - t4_r * S1;
    T m4_i = t2_i * S2 - t4_i * S1;

    // Combine results
    // Out 1: m1 - j*m3
    out[1*stride_out]   = m1_r + m3_i; 
    out[1*stride_out+1] = m1_i - m3_r;

    // Out 2: m2 - j*m4
    out[2*stride_out]   = m2_r + m4_i;
    out[2*stride_out+1] = m2_i - m4_r;

    // Out 3: m2 + j*m4
    out[3*stride_out]   = m2_r - m4_i;
    out[3*stride_out+1] = m2_i + m4_r;

    // Out 4: m1 + j*m3
    out[4*stride_out]   = m1_r - m3_i;
    out[4*stride_out+1] = m1_i + m3_r;
}

template <typename T>
inline void Radix6_Butterfly
(
    const T* in,
    T* out,
    int32_t stride_in,
    int32_t stride_out
) noexcept
{
    // Constants for Radix-6
    // Based on PI/3 = 60 degrees
    // W6^1 = 0.5 - j*sqrt(3)/2
    // W6^2 = -0.5 - j*sqrt(3)/2
    // Radix-3 constants match these values.
    
    constexpr T HALF  = static_cast<T>(0.5);
    constexpr T SIN60 = static_cast<T>(0.8660254037844386);

    // ======================================================================
    // LOAD INPUTS (Unrolled)
    // ======================================================================
    // Logic: 3 Columns of 2
    // Col 0: 0, 3
    // Col 1: 1, 4
    // Col 2: 2, 5
    
    T r0 = in[0];             T i0 = in[1];
    T r1 = in[stride_in];     T i1 = in[stride_in+1];
    T r2 = in[2*stride_in];   T i2 = in[2*stride_in+1];
    T r3 = in[3*stride_in];   T i3 = in[3*stride_in+1];
    T r4 = in[4*stride_in];   T i4 = in[4*stride_in+1];
    T r5 = in[5*stride_in];   T i5 = in[5*stride_in+1];

    // ======================================================================
    // STAGE 1: COLUMNS (3x Radix-2)
    // ======================================================================
    // Calculate Sums (Row 0) and Diffs (Row 1)
    
    // Col 0
    T t0_r = r0 + r3; T t0_i = i0 + i3; // Row 0, Col 0
    T t3_r = r0 - r3; T t3_i = i0 - i3; // Row 1, Col 0 (No Twiddle W^0)

    // Col 1
    T t1_r = r1 + r4; T t1_i = i1 + i4; // Row 0, Col 1
    T d1_r = r1 - r4; T d1_i = i1 - i4; // Row 1, Col 1 (Needs W^1)

    // Col 2
    T t2_r = r2 + r5; T t2_i = i2 + i5; // Row 0, Col 2
    T d2_r = r2 - r5; T d2_i = i2 - i5; // Row 1, Col 2 (Needs W^2)

    // ======================================================================
    // STAGE 2: TWIDDLES (W_6)
    // ======================================================================
    
    // Apply W^1 to d1 (0.5 - j*SIN60)
    // (r + ji)(0.5 - jS) = 0.5r - jSr + 0.5ji + S
    // Re = 0.5r + S*i
    // Im = 0.5i - S*r
    T t4_r = d1_r * HALF + d1_i * SIN60;
    T t4_i = d1_i * HALF - d1_r * SIN60;

    // Apply W^2 to d2 (-0.5 - j*SIN60)
    // (r + ji)(-0.5 - jS)
    // Re = -0.5r + S*i
    // Im = -0.5i - S*r
    T t5_r = d2_r * -HALF + d2_i * SIN60;
    T t5_i = d2_i * -HALF - d2_r * SIN60;

    // ======================================================================
    // STAGE 3: ROWS (2x Radix-3) & OUTPUT
    // ======================================================================
    // Row 0 Inputs: t0, t1, t2 -> Maps to Evens (Indices 0, 2, 4)
    // Row 1 Inputs: t3, t4, t5 -> Maps to Odds  (Indices 1, 3, 5)

    // --- ROW 0 (Radix-3) ---
    {
        T sum_r = t1_r + t2_r;
        T sum_i = t1_i + t2_i;
        
        // Output 0 (DC)
        out[0] = t0_r + sum_r;
        out[1] = t0_i + sum_i;
        
        // Core Radix-3 math
        T m1_r = t0_r - sum_r * HALF;
        T m1_i = t0_i - sum_i * HALF;
        T m2_r = (t1_r - t2_r) * SIN60;
        T m2_i = (t1_i - t2_i) * SIN60;
        
        // Output 2
        out[2*stride_out]   = m1_r + m2_i;
        out[2*stride_out+1] = m1_i - m2_r;
        
        // Output 4
        out[4*stride_out]   = m1_r - m2_i;
        out[4*stride_out+1] = m1_i + m2_r;
    }

    // --- ROW 1 (Radix-3) ---
    {
        T sum_r = t4_r + t5_r;
        T sum_i = t4_i + t5_i;
        
        // Output 1
        out[1*stride_out]   = t3_r + sum_r;
        out[1*stride_out+1] = t3_i + sum_i;
        
        T m1_r = t3_r - sum_r * HALF;
        T m1_i = t3_i - sum_i * HALF;
        T m2_r = (t4_r - t5_r) * SIN60;
        T m2_i = (t4_i - t5_i) * SIN60;
        
        // Output 3
        out[3*stride_out]   = m1_r + m2_i;
        out[3*stride_out+1] = m1_i - m2_r;
        
        // Output 5
        out[5*stride_out]   = m1_r - m2_i;
        out[5*stride_out+1] = m1_i + m2_r;
    }
    
    return;
}

template <typename T>
inline void Radix7_Butterfly
(
	const T* in,
	T* out,
	int32_t stride_in,
	int32_t stride_out
) noexcept
{
    // Constants (2*pi/7 based)
    // u = 2pi/7
    // c1=cos(u), c2=cos(2u), c3=cos(3u)
    // s1=sin(u), s2=sin(2u), s3=sin(3u)
    
    constexpr T C1 = static_cast<T>(0.6234898018587335);
    constexpr T C2 = static_cast<T>(-0.2225209339563144);
    constexpr T C3 = static_cast<T>(-0.9009688679024191);
    constexpr T S1 = static_cast<T>(0.7818314824680298);
    constexpr T S2 = static_cast<T>(0.9749279121818236);
    constexpr T S3 = static_cast<T>(0.4338837391175581);

    CACHE_ALIGN T r[7];
    CACHE_ALIGN T i[7];
    
    // Load inputs
    for(int32_t k = 0; k < 7; ++k)
    {
        r[k] = in[k*stride_in];
        i[k] = in[k*stride_in+1];
    }

    // Sums and Diffs
    T t1_r = r[1] + r[6]; T t1_i = i[1] + i[6];
    T t2_r = r[1] - r[6]; T t2_i = i[1] - i[6];
    T t3_r = r[2] + r[5]; T t3_i = i[2] + i[5];
    T t4_r = r[2] - r[5]; T t4_i = i[2] - i[5];
    T t5_r = r[3] + r[4]; T t5_i = i[3] + i[4];
    T t6_r = r[3] - r[4]; T t6_i = i[3] - i[4];

    // DC Component
    out[0] = r[0] + t1_r + t3_r + t5_r;
    out[1] = i[0] + t1_i + t3_i + t5_i;

    // Real parts mix (using Cosines)
    T m1 = r[0] + t1_r*C1 + t3_r*C2 + t5_r*C3;
    T m2 = r[0] + t1_r*C2 + t3_r*C3 + t5_r*C1;
    T m3 = r[0] + t1_r*C3 + t3_r*C1 + t5_r*C2;
    
    T n1 = i[0] + t1_i*C1 + t3_i*C2 + t5_i*C3;
    T n2 = i[0] + t1_i*C2 + t3_i*C3 + t5_i*C1;
    T n3 = i[0] + t1_i*C3 + t3_i*C1 + t5_i*C2;

    // Imag parts mix (using Sines)
    T p1 = t2_r*S1 + t4_r*S2 + t6_r*S3;
    T p2 = t2_r*S2 - t4_r*S3 - t6_r*S1;
    T p3 = t2_r*S3 - t4_r*S1 + t6_r*S2;

    T q1 = t2_i*S1 + t4_i*S2 + t6_i*S3;
    T q2 = t2_i*S2 - t4_i*S3 - t6_i*S1;
    T q3 = t2_i*S3 - t4_i*S1 + t6_i*S2;

    // Recombine
    // k=1: m1 - j*q1, p1 (Real part uses p1? No, logic: Re = m - q_imag, Im = n + p_real)
    // (a+bi)(c+di) logic is simpler.
    // Re[1] = m1 + q1; Im[1] = n1 - p1; // Wait, sign convention?
    
    // Standard DFT mapping:
    // Out[1] = m1 - i*q1 is wrong. 
    // Let's stick to sums:
    // Out[1] = (m1 + q1) + j(n1 - p1)
    // Out[6] = (m1 - q1) + j(n1 + p1)
    
    // Out 1
    out[1*stride_out]   = m1 + q1; 
    out[1*stride_out+1] = n1 - p1;
    // Out 6
    out[6*stride_out]   = m1 - q1; 
    out[6*stride_out+1] = n1 + p1;

    // Out 2
    out[2*stride_out]   = m2 + q2; 
    out[2*stride_out+1] = n2 - p2;
    // Out 5
    out[5*stride_out]   = m2 - q2; 
    out[5*stride_out+1] = n2 + p2;

    // Out 3
    out[3*stride_out]   = m3 + q3; 
    out[3*stride_out+1] = n3 - p3;
    // Out 4
    out[4*stride_out]   = m3 - q3; 
    out[4*stride_out+1] = n3 + p3;
}

template <typename T>
inline void Radix8_Butterfly
(
    const T* in,
    T* out,
    int32_t stride_in,
    int32_t stride_out
) noexcept
{
    // Constants
    constexpr T SQRT2_2 = static_cast<T>(0.7071067811865475);

    // Load Inputs (0..7)
    T r0 = in[0];               T i0 = in[1];
    T r1 = in[stride_in];       T i1 = in[stride_in+1];
    T r2 = in[2*stride_in];     T i2 = in[2*stride_in+1];
    T r3 = in[3*stride_in];     T i3 = in[3*stride_in+1];
    T r4 = in[4*stride_in];     T i4 = in[4*stride_in+1];
    T r5 = in[5*stride_in];     T i5 = in[5*stride_in+1];
    T r6 = in[6*stride_in];     T i6 = in[6*stride_in+1];
    T r7 = in[7*stride_in];     T i7 = in[7*stride_in+1];

    // Stage 1 (DIF: Add/Sub first half with second half)
    T t0_r = r0 + r4; T t0_i = i0 + i4;
    T t1_r = r0 - r4; T t1_i = i0 - i4;
    T t2_r = r1 + r5; T t2_i = i1 + i5;
    T t3_r = r1 - r5; T t3_i = i1 - i5;
    T t4_r = r2 + r6; T t4_i = i2 + i6;
    T t5_r = r2 - r6; T t5_i = i2 - i6;
    T t6_r = r3 + r7; T t6_i = i3 + i7;
    T t7_r = r3 - r7; T t7_i = i3 - i7;

    // Stage 2 (Twiddles & Sub-Butterflies)
    // Group 0 (Evens): t0, t2, t4, t6
    // Group 1 (Odds):  t1, t3, t5, t7
    
    // -- Process Evens (effectively Radix-4 on 0,2,4,6) --
    // Butterfly (t0, t4) and (t2, t6)
    // Note: t4/t6 need W_4 factors (1, -j) relative to group? No, standard DIF recursive logic:
    // S2_0 = t0 + t4
    // S2_1 = t0 - t4
    // S2_2 = t2 + t6
    // S2_3 = (t2 - t6) * -j
    
    T ev0_r = t0_r + t4_r; T ev0_i = t0_i + t4_i;
    T ev1_r = t0_r - t4_r; T ev1_i = t0_i - t4_i;
    T ev2_r = t2_r + t6_r; T ev2_i = t2_i + t6_i;
    T ev3_r = t2_i - t6_i; T ev3_i = t6_r - t2_r; // (t2 - t6)*(-j) => r=i_diff, i=-r_diff

    // -- Process Odds (t1, t3, t5, t7) --
    // Must apply W_8 factors FIRST
    // t1 * 1
    // t3 * W^1 (0.707 - j0.707)
    // t5 * W^2 (-j)
    // t7 * W^3 (-0.707 - j0.707)

    // t3_new
    T temp_r = (t3_r + t3_i) * SQRT2_2;
    T temp_i = (t3_i - t3_r) * SQRT2_2;
    t3_r = temp_r; t3_i = temp_i;

    // t5_new = t5 * -j
    temp_r = t5_i; 
    t5_i = -t5_r; 
    t5_r = temp_r;

    // t7_new
    // W^3 = -0.707 - j0.707 = -SQRT2_2 * (1 + j)
    temp_r = (t7_i - t7_r) * SQRT2_2;
    temp_i = -(t7_r + t7_i) * SQRT2_2;
    t7_r = temp_r; t7_i = temp_i;

    // Now Radix-4 Butterfly on Odds
    T od0_r = t1_r + t5_r; T od0_i = t1_i + t5_i;
    T od1_r = t1_r - t5_r; T od1_i = t1_i - t5_i;
    T od2_r = t3_r + t7_r; T od2_i = t3_i + t7_i;
    T od3_r = t3_r - t7_r; T od3_i = t3_i - t7_i; // Wait, missing -j on second half?
    // In Radix-4 sub-stage:
    // Top: (t1+t5), (t3+t7)
    // Bot: (t1-t5), (t3-t7)*(-j)
    // Let's re-do odd butterfly cleanly:
    
    // Butterfly 1
    T o1_r = t1_r + t5_r; T o1_i = t1_i + t5_i;
    T o2_r = t1_r - t5_r; T o2_i = t1_i - t5_i;
    
    // Butterfly 2
    T o3_r = t3_r + t7_r; T o3_i = t3_i + t7_i;
    T o4_r = t3_r - t7_r; T o4_i = t3_i - t7_i;
    // Apply -j to o4 (DIF internal twiddle)
    T tmp = o4_r; o4_r = o4_i; o4_i = -tmp;

    // Stage 3 (Final Mapping to Natural Order)
    // Even Group -> Outputs 0, 2, 4, 6
    // Odd Group  -> Outputs 1, 3, 5, 7

    // Evens: Radix-2 on (ev0, ev2) and (ev1, ev3)
    // X[0] = ev0 + ev2
    out[0]               = ev0_r + ev2_r; out[1]               = ev0_i + ev2_i;
    // X[4] = ev0 - ev2
    out[4*stride_out]    = ev0_r - ev2_r; out[4*stride_out+1]  = ev0_i - ev2_i;
    // X[2] = ev1 + ev3
    out[2*stride_out]    = ev1_r + ev2_r; // Wait. Logic check.
    // ev0 is sum(0,4), ev2 is sum(2,6). -> 0+4+2+6 = X[0]. Correct.
    // ev1 is diff(0,4), ev3 is diff(2,6)*-j.
    // X[2] should be (x0-x4) + (x2-x6)*(-1)? No.
    // Let's rely on standard DIF output order for 4-point: 0, 2, 1, 3.
    // So Evens produce: 0->0, 1->2, 2->1(which maps to 2), 3->3(maps to 6).
    // Let's compute explicitly:
    
    // X[2]
    out[2*stride_out]    = ev1_r + ev3_r; out[2*stride_out+1]  = ev1_i + ev3_i;
    // X[6]
    out[6*stride_out]    = ev1_r - ev3_r; out[6*stride_out+1]  = ev1_i - ev3_i;

    // Odds: Radix-2 on (o1, o3) and (o2, o4)
    // X[1]
    out[1*stride_out]    = o1_r + o3_r; out[1*stride_out+1]    = o1_i + o3_i;
    // X[5]
    out[5*stride_out]    = o1_r - o3_r; out[5*stride_out+1]    = o1_i - o3_i;
    // X[3]
    out[3*stride_out]    = o2_r + o4_r; out[3*stride_out+1]    = o2_i + o4_i;
    // X[7]
    out[7*stride_out]    = o2_r - o4_r; out[7*stride_out+1]    = o2_i - o4_i;
    
    return;
}

template <typename T>
inline void Radix9_Butterfly
(
    const T* in,
    T* out,
    int32_t stride_in,
    int32_t stride_out
) noexcept
{
    constexpr T C40 = static_cast<T>(0.7660444431189780); 
    constexpr T S40 = static_cast<T>(0.6427876096865393); 
    constexpr T C80 = static_cast<T>(0.1736481776669303); 
    constexpr T S80 = static_cast<T>(0.9848077530122080); 
    constexpr T C160 = static_cast<T>(-0.9396926207859084); 
    constexpr T S160 = static_cast<T>(0.3420201433256687);  
    constexpr T R3_C = static_cast<T>(-0.5);
    constexpr T R3_S = static_cast<T>(0.8660254037844386);

    // Intermediate Buffer (3 cols x 3 rows)
    CACHE_ALIGN T t[18]; 

    // ------------------------------------------------------------------
    // STAGE 1: COLUMNS (3x Radix-3)
    // ------------------------------------------------------------------
    // Input Mapping:
    // Col 0: in[0], in[3], in[6]
    // Col 1: in[1], in[4], in[7]
    // Col 2: in[2], in[5], in[8]
    
    // Process Col 0 -> t[0], t[1], t[2] (Complex indices)
    {
        T r0 = in[0];             T i0 = in[1];
        T r1 = in[3*stride_in];   T i1 = in[3*stride_in+1];
        T r2 = in[6*stride_in];   T i2 = in[6*stride_in+1];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        t[0] = r0 + t1; t[1] = i0 + t2; // Out 0
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        t[2] = m1 + m4; t[3] = m2 - m3; // Out 1
        t[4] = m1 - m4; t[5] = m2 + m3; // Out 2
    }

    // Process Col 1 -> t[3], t[4], t[5]
    {
        T r0 = in[1*stride_in];   T i0 = in[1*stride_in+1];
        T r1 = in[4*stride_in];   T i1 = in[4*stride_in+1];
        T r2 = in[7*stride_in];   T i2 = in[7*stride_in+1];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        t[6] = r0 + t1; t[7] = i0 + t2;
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        t[8] = m1 + m4; t[9] = m2 - m3;
        t[10] = m1 - m4; t[11] = m2 + m3;
    }

    // Process Col 2 -> t[6], t[7], t[8]
    {
        T r0 = in[2*stride_in];   T i0 = in[2*stride_in+1];
        T r1 = in[5*stride_in];   T i1 = in[5*stride_in+1];
        T r2 = in[8*stride_in];   T i2 = in[8*stride_in+1];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        t[12] = r0 + t1; t[13] = i0 + t2;
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        t[14] = m1 + m4; t[15] = m2 - m3;
        t[16] = m1 - m4; t[17] = m2 + m3;
    }

    // ------------------------------------------------------------------
    // STAGE 2: TWIDDLES (In-Place on 't')
    // ------------------------------------------------------------------
    // t[0..5] is Col 0 Result (No Twiddle)
    // t[6..11] is Col 1 Result
    // t[12..17] is Col 2 Result
    
    // Apply W9 factors to the "Rows" of the intermediate matrix.
    // Row 1 (Index 1 of each col result): t[2], t[8], t[14]
    // Row 2 (Index 2 of each col result): t[4], t[10], t[16]
    
    // Col 1, Row 1 (t[8]) -> W^1
    CMul(t[8], t[9], C40, -S40);
    // Col 1, Row 2 (t[10]) -> W^2
    CMul(t[10], t[11], C80, -S80);
    
    // Col 2, Row 1 (t[14]) -> W^2
    CMul(t[14], t[15], C80, -S80);
    // Col 2, Row 2 (t[16]) -> W^4
    CMul(t[16], t[17], C160, -S160);

    // ------------------------------------------------------------------
    // STAGE 3: ROWS (3x Radix-3) & NATURAL OUTPUT
    // ------------------------------------------------------------------
    // We process Rows of 't'.
    // Row 0 Inputs: t[0], t[6], t[12]  -> Maps to Output 0, 3, 6 (Stride 3)
    // Row 1 Inputs: t[2], t[8], t[14]  -> Maps to Output 1, 4, 7
    // Row 2 Inputs: t[4], t[10], t[16] -> Maps to Output 2, 5, 8
    
    // Row 0
    {
        T r0 = t[0]; T i0 = t[1];
        T r1 = t[6]; T i1 = t[7];
        T r2 = t[12]; T i2 = t[13];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        // Output 0
        out[0] = r0 + t1; 
        out[1] = i0 + t2;
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        // Output 3
        out[3*stride_out] = m1 + m4; 
        out[3*stride_out+1] = m2 - m3;
        
        // Output 6
        out[6*stride_out] = m1 - m4; 
        out[6*stride_out+1] = m2 + m3;
    }

    // Row 1
    {
        T r0 = t[2]; T i0 = t[3];
        T r1 = t[8]; T i1 = t[9];
        T r2 = t[14]; T i2 = t[15];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        // Output 1
        out[1*stride_out] = r0 + t1; 
        out[1*stride_out+1] = i0 + t2;
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        // Output 4
        out[4*stride_out] = m1 + m4; 
        out[4*stride_out+1] = m2 - m3;
        
        // Output 7
        out[7*stride_out] = m1 - m4; 
        out[7*stride_out+1] = m2 + m3;
    }

    // Row 2
    {
        T r0 = t[4]; T i0 = t[5];
        T r1 = t[10]; T i1 = t[11];
        T r2 = t[16]; T i2 = t[17];
        
        T t1 = r1 + r2; T t2 = i1 + i2;
        // Output 2
        out[2*stride_out] = r0 + t1; 
        out[2*stride_out+1] = i0 + t2;
        
        T m1 = r0 + t1 * R3_C; T m2 = i0 + t2 * R3_C;
        T m3 = (r1 - r2) * R3_S; T m4 = (i1 - i2) * R3_S;
        
        // Output 5
        out[5*stride_out] = m1 + m4; 
        out[5*stride_out+1] = m2 - m3;
        
        // Output 8
        out[8*stride_out] = m1 - m4; 
        out[8*stride_out+1] = m2 + m3;
    }
}


template <typename T>
inline void Radix16_Butterfly
(
    const T* in,
    T* out,
    int32_t stride
) noexcept
{
    // Constants
    constexpr T C1 = static_cast<T>(0.9238795325112867); // cos(pi/8)
    constexpr T S1 = static_cast<T>(0.3826834323650898); // sin(pi/8)
    constexpr T C2 = static_cast<T>(0.7071067811865475); // cos(pi/4)
    constexpr T S2 = static_cast<T>(0.7071067811865475); // sin(pi/4)
    constexpr T C3 = static_cast<T>(0.3826834323650898); // cos(3pi/8)
    constexpr T S3 = static_cast<T>(0.9238795325112867); // sin(3pi/8)

    // Intermediate buffer
    CACHE_ALIGN T t[32];

    // =======================================================================
    // STAGE 1: COLUMNS (4x Radix-4)
    // =======================================================================
    // We treat the input as a 4x4 Matrix.
    // We process Columns 0, 1, 2, 3.
    // Stride between row elements in input is 8 floats (4 complex).
    
    // Column 0 Inputs: 0, 4, 8, 12
    Radix4_Butterfly(in + 0, t + 0, 8, 2); 
    // Column 1 Inputs: 1, 5, 9, 13
    Radix4_Butterfly(in + 2, t + 8, 8, 2); 
    // Column 2 Inputs: 2, 6, 10, 14
    Radix4_Butterfly(in + 4, t + 16, 8, 2); 
    // Column 3 Inputs: 3, 7, 11, 15
    Radix4_Butterfly(in + 6, t + 24, 8, 2); 

    // =======================================================================
    // STAGE 2: TWIDDLES
    // =======================================================================
    // 't' now holds [Col0, Col1, Col2, Col3] sequentially.
    // We need to apply W_16 factors to the "Rows" of this result matrix.
    // Row 0 corresponds to elements 0 of each block (indices 0, 4, 8, 12 in 't' complex terms).
    
    // Row 0: Multiplied by 1 (No Op)
    
    // Row 1: Elements at indices 1, 5, 9, 13 (complex) in 't'
    // Multipliers: W^0, W^1, W^2, W^3
    // t[2,3] is Col0,Row1 -> W^0 (Skip)
    CMul(t[10], t[11], C1, -S1); // Col1,Row1 -> W^1
    CMul(t[18], t[19], C2, -S2); // Col2,Row1 -> W^2
    CMul(t[26], t[27], C3, -S3); // Col3,Row1 -> W^3

    // Row 2: Elements at indices 2, 6, 10, 14 (complex) in 't'
    // Multipliers: W^0, W^2, W^4, W^6
    // t[4,5] is Col0,Row2 -> W^0
    CMul(t[12], t[13], C2, -S2); // Col1,Row2 -> W^2
    { T tr = t[20]; t[20] = t[21]; t[21] = -tr; } // Col2,Row2 -> W^4 = -j
    CMul(t[28], t[29], -C2, -S2); // Col3,Row2 -> W^6

    // Row 3: Elements at indices 3, 7, 11, 15 (complex) in 't'
    // Multipliers: W^0, W^3, W^6, W^9
    // t[6,7] is Col0,Row3 -> W^0
    CMul(t[14], t[15], C3, -S3); // Col1,Row3 -> W^3
    CMul(t[22], t[23], -C2, -S2); // Col2,Row3 -> W^6
    CMul(t[30], t[31], -C1, S1);  // Col3,Row3 -> W^9 (-C1 + jS1)

    // =======================================================================
    // STAGE 3: ROWS (4x Radix-4) & TRANSPOSE MAPPING
    // =======================================================================
    // We perform Radix-4 on the Rows.
    // Row 0 inputs are: t[0], t[8], t[16], t[24] (Floats).
    // Wait! 't' is complex interleaved.
    // Col0 is t[0..7]. Col1 is t[8..15].
    // Row 0 is the 0th complex element of each Col block.
    // So inputs are: t[0], t[8], t[16], t[24].
    // Stride is 8 floats (4 complex).
    
    // We write to a temporary buffer 't2' to keep it contiguous for the final map.
    CACHE_ALIGN T t2[32];

    // Process Row 0 -> t2[0..7]
    Radix4_Butterfly(t + 0, t2 + 0, 8, 2);
    // Process Row 1 -> t2[8..15]
    // Inputs start at t[2] (1st complex element of Col0)
    Radix4_Butterfly(t + 2, t2 + 8, 8, 2);
    // Process Row 2 -> t2[16..23]
    Radix4_Butterfly(t + 4, t2 + 16, 8, 2);
    // Process Row 3 -> t2[24..31]
    Radix4_Butterfly(t + 6, t2 + 24, 8, 2);

    // =======================================================================
    // STAGE 4: OUTPUT MAPPING (Transpose)
    // =======================================================================
    // 't2' contains [RowOutput0, RowOutput1, RowOutput2, RowOutput3]
    // But the true Frequency Index is: k = 4*k2 + k1
    // where k1 is the Row Index (0..3) and k2 is the index within the Row Output.
    // This means the results are effectively Transposed.
    // Row 0 (k1=0) contains outputs for Total Index: 0, 4, 8, 12.
    // Row 1 (k1=1) contains outputs for Total Index: 1, 5, 9, 13.
    
    // We manually assign to 'out' to ensure perfect ordering 0..15
    
    // From Row 0 Result (t2[0..7]) -> Dest 0, 4, 8, 12
    out[0] = t2[0]; out[1] = t2[1];   // Dest 0
    out[8] = t2[2]; out[9] = t2[3];   // Dest 4
    out[16]= t2[4]; out[17]= t2[5];   // Dest 8
    out[24]= t2[6]; out[25]= t2[7];   // Dest 12
    
    // From Row 1 Result (t2[8..15]) -> Dest 1, 5, 9, 13
    out[2] = t2[8]; out[3] = t2[9];   // Dest 1
    out[10]= t2[10];out[11]= t2[11];  // Dest 5
    out[18]= t2[12];out[19]= t2[13];  // Dest 9
    out[26]= t2[14];out[27]= t2[15];  // Dest 13

    // From Row 2 Result (t2[16..23]) -> Dest 2, 6, 10, 14
    out[4] = t2[16];out[5] = t2[17];  // Dest 2
    out[12]= t2[18];out[13]= t2[19];  // Dest 6
    out[20]= t2[20];out[21]= t2[21];  // Dest 10
    out[28]= t2[22];out[29]= t2[23];  // Dest 14

    // From Row 3 Result (t2[24..31]) -> Dest 3, 7, 11, 15
    out[6] = t2[24];out[7] = t2[25];  // Dest 3
    out[14]= t2[26];out[15]= t2[27];  // Dest 7
    out[22]= t2[28];out[23]= t2[29];  // Dest 11
    out[30]= t2[30];out[31]= t2[31];  // Dest 15
}


template <typename T>
inline void unshuffle_mixed_radix
(
    T* dst,
    int32_t N,
    const std::vector<int32_t>& factors
) noexcept
{
    std::vector<T> temp(2 * N);
    
    // 1. Calculate Weights for the "Reversed" Mixed-Radix System
    // We want to map the digit extracted from the Last Factor to the Most Significant Position.
    //
    // Example Factors: {16, 2}
    // We extract digits d_2 (from 2) and d_16 (from 16).
    // Target Index = d_2 * 16 + d_16 * 1.
    //
    // General Algorithm:
    // We iterate factors backwards (as we extract digits).
    // The weight for the current digit is the product of all *remaining* factors 
    // in the forward list (or accumulated product of previously processed reversed factors).
    
    // Let's build weights corresponding to the iterators of 'factors.rbegin()'.
    std::vector<int32_t> reversed_weights;
    reversed_weights.reserve(factors.size());
    
    int32_t current_weight = 1;
    // Iterate Forward to build weights [1, 16] for factors [16, 2]??
    // No.
    // If factors are [A, B, C].
    // Digit from C (last) has weight (A*B).
    // Digit from B has weight (A).
    // Digit from A has weight (1).
    
    // Let's pre-calculate these specific weights.
    // We can compute them by traversing the factor list Forward.
    // Weights: [1] -> [1, A] -> [1, A, A*B]... 
    // But we need to match them to the Reverse Iterator (C, then B, then A).
    // So we need the weights in order: (A*B), (A), (1).
    
    // Step A: Build Standard Weights [1, C, B*C, A*B*C...] (Little Endian)
    // No, we want Big Endian weights? 
    // Let's simply generate the weights list [ (N/F0), (N/F0*F1), ... 1 ]
    // Then reverse it to match the reverse iteration of factors.
    
    std::vector<int32_t> target_weights;
    int32_t w = N;
    for (auto const& radix : factors)
    {
        w /= radix;
        target_weights.push_back(w);
    }
    // Factors {16, 2} -> Weights {2, 1}.
    // We want to map Last Factor (2) to First Weight (Wait, which one is MSB?)
    // In Bit Reversal: LSB becomes MSB.
    // Last Factor is LSB (finest grain).
    // So Last Factor digit should get the MSB Weight.
    // MSB Weight is the First one in 'target_weights' (value 16).
    
    // So: Reverse Iterator [2, 16] should match Weights [16, 1]?
    // target_weights is [2, 1].
    // We want [16, 1].
    // How to get [16, 1]?
    // 16 = N/2.
    // 1 = N/(2*16).
    
    // Correct Logic:
    // The weight for the digit extracted from 'radix' is (N / cumulative_product_so_far * radix) ?
    // Simpler:
    // Weight for Factor[i] (in reverse list) is Product(Factors[0]...Factors[i-1]).
    // Factor 0 (Last in input): 2. Weight should be 16 (Product of others).
    // Factor 1 (First in input): 16. Weight should be 1.
    
    // Let's build the correct weights dynamically.
    
    std::vector<int32_t> correct_weights;
    int32_t product_so_far = 1;
    // Iterate Forward through factors to build the weights for the Reversed Digits.
    // If Factors = {16, 2}.
    // Reversal = {2, 16}.
    // Weight for 2 should be 16.
    // Weight for 16 should be 1.
    // This is simply: weights[k] = N / (product_so_far * current_reversed_factor).
    
    // Let's use the list of factors reversed: {2, 16}.
    std::vector<int32_t> factors_reversed = factors;
    std::reverse(factors_reversed.begin(), factors_reversed.end());
    
    int32_t running_product = 1;
    for (int32_t r : factors_reversed)
    {
        // We want the weight to fill the REMAINING space.
        // For {2, 16}:
        // r=2. We want weight 16. 
        // 16 = N / 2.
        // r=16. We want weight 1.
        // 1 = N / (2*16).
        correct_weights.push_back(N / (running_product * r));
        running_product *= r;
    }

    // 2. Permutation Loop
    for (int32_t i = 0; i < N; ++i)
    {
        int32_t input_idx = i;
        int32_t target_idx = 0;
        
        // Iterate Factors Reversed
        for (size_t k = 0; k < factors_reversed.size(); ++k)
        {
            int32_t R = factors_reversed[k];
            
            int32_t digit = input_idx % R;
            input_idx /= R;
            
            target_idx += digit * correct_weights[k];
        }
        
        temp[2 * target_idx]     = dst[2 * i];
        temp[2 * target_idx + 1] = dst[2 * i + 1];
    }

    std::memcpy(dst, temp.data(), 2 * N * sizeof(T));
}


// ============================================================================
// ITERATIVE MIXED-RADIX (Out-Of-Place Optimized)
// ============================================================================
// Strategy: 
// 1. First Stage: Read 'src' -> Process -> Write 'dst'. (Breaks Aliasing)
// 2. Next Stages: Read 'dst' -> Process -> Write 'dst'.
// 3. Unshuffle:   Permute 'dst'.
// ----------------------------------------------------------------------------
template <typename T>
void FFT_MixedRadix_Iterative (const T* src, T* dst, int32_t N, const std::vector<int32_t>& factors) noexcept
{
    int32_t group_size = N;
    int32_t block_count = 1;
    
    // Flag to track the very first pass
    bool is_first_stage = true;

    // --- STAGE LOOP ---
    for (auto const& R : factors)
	{
        const int32_t stride = group_size / R;

        // --- BLOCK LOOP ---
        for (int32_t b = 0; b < block_count; ++b)
		{
            const int32_t base_offset = b * group_size;

            // --- BUTTERFLY LOOP ---
            for (int32_t k = 0; k < stride; ++k)
			{
                
                // 1. Gather inputs into local stack buffer
                // OPTIMIZATION: 
                // If this is the first stage, we read from the read-only 'src'.
                // This allows the compiler to generate non-aliased loads.
                // If it's a later stage, the data is already in 'dst', so we read from there.
                
                CACHE_ALIGN T local_buf[32]; // Max Radix 16 * 2 floats
                
                if (is_first_stage)
				{
                    // LOAD FROM SOURCE
                    for (int32_t j = 0; j < R; ++j)
					{
                        int32_t idx = 2 * (base_offset + k + j * stride);
                        local_buf[2 * j]     = src[idx];
                        local_buf[2 * j + 1] = src[idx + 1];
                    }
                } else
				{
                    // LOAD FROM DESTINATION (Intermediate result)
                    for (int32_t j = 0; j < R; ++j)
					{
                        int32_t idx = 2 * (base_offset + k + j * stride);
                        local_buf[2 * j]     = dst[idx];
                        local_buf[2 * j + 1] = dst[idx + 1];
                    }
                }

                // 2. Dispatch Butterfly (Compute in Registers/L1)
                // Stride is 2 (contiguous complex) inside local_buf.
                switch (R)
				{
                    case 16: FourierTransform::Radix16_Butterfly(local_buf, local_buf, 2); break;
                    case 9:  FourierTransform::Radix9_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 8:  FourierTransform::Radix8_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 7:  FourierTransform::Radix7_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 6:  FourierTransform::Radix6_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 5:  FourierTransform::Radix5_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 4:  FourierTransform::Radix4_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 3:  FourierTransform::Radix3_Butterfly(local_buf, local_buf, 2, 2); break;
                    case 2:  FourierTransform::Radix2_Butterfly(local_buf, local_buf, 2, 2); break;
                }

                // 3. Apply Twiddle & Store
                // ALWAYS WRITE TO DESTINATION
                
                // Row 0 (W^0 = 1)
                int32_t idx0 = 2 * (base_offset + k);
                dst[idx0]     = local_buf[0];
                dst[idx0 + 1] = local_buf[1];

                // Rows 1..R-1
                for (int32_t j = 1; j < R; ++j)
				{
                    T& r = local_buf[2 * j];
                    T& i = local_buf[2 * j + 1];
                    
                    // Twiddle
                    FourierTransform::apply_twiddle (r, i, j * k, group_size);

                    int32_t idx = 2 * (base_offset + k + j * stride);
                    dst[idx]     = r;
                    dst[idx + 1] = i;
                }
            }
        }
        
        // After the first loop completes, all data is now in 'dst'.
        // All subsequent loops must read from 'dst'.
        is_first_stage = false;
        
        // Update state for next stage
        block_count *= R;
        group_size /= R;
    }

    // Final permutation on the output buffer
    FourierTransform::unshuffle_mixed_radix(dst, N, factors);
    return;
}
    
}