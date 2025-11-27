#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace FourierTransform
{

constexpr int32_t cztPerformanceThreshold = 90;
	
// ============================================================================
// PART 1: HELPER UTILITIES
// ============================================================================

// Calculates the smallest power of 2 >= n
inline int32_t NextPowerOf2 (int32_t n) noexcept
{
    if (n < 1) return 1;
    int32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ----------------------------------------------------------------------------
// MOCK POW-2 FFT (REPLACE THIS WITH YOUR FAST RADIX-16 ENGINE)
// ----------------------------------------------------------------------------
// This simplistic recursive implementation is here ONLY to make the CZT code 
// below runnable in this snippet. Do not use this function in production.
template <typename T>
inline void FFT_Pow2_Helper (int32_t N, T* data, bool inverse)
{
    if (N <= 1) return;
    
    // Separate even/odd
    const int32_t M = N / 2;
    std::vector<T> even(2 * M);
    std::vector<T> odd(2 * M);
    
    for (int i = 0; i < M; ++i)
	{
        even[2*i]   = data[4*i];   even[2*i+1] = data[4*i+1];
        odd[2*i]    = data[4*i+2]; odd[2*i+1] = data[4*i+3];
    }
    
    FFT_Pow2_Helper(M, even.data(), inverse);
    FFT_Pow2_Helper(M, odd.data(), inverse);
    
    constexpr T PI {static_cast<T>(3.14159265358979323846)};
    const T angle_dir = inverse ? static_cast<T>(2.0) * PI : static_cast<T>(-2.0) * PI;
    
    for (int k = 0; k < M; ++k)
	{
        // W = exp(j * angle * k / N)
        const T theta = angle_dir * static_cast<T>(k) / static_cast<T>(N);
        T wr = static_cast<T>(std::cos(theta));
        T wi = static_cast<T>(std::sin(theta));
        
        // Complex Mult: Odd[k] * W
        T or_val = odd[2*k];
        T oi_val = odd[2*k+1];
        T tr = or_val * wr - oi_val * wi;
        T ti = or_val * wi + oi_val * wr;
        
        T er = even[2*k];
        T ei = even[2*k+1];
        
        data[2*k]     = er + tr;
        data[2*k+1]   = ei + ti;
        data[2*(k+M)]   = er - tr;
        data[2*(k+M)+1] = ei - ti;
    }
}

// ----------------------------------------------------------------------------
// POWER-OF-2 FFT INTERFACE (Use your engine here)
// ----------------------------------------------------------------------------
template <typename T>
inline void Execute_FFT_Pow2 (int32_t M, T* data)
{
    // In your real code: Execute_FFT_Iterative(M, data, {16, 16, 4...});
    FFT_Pow2_Helper (M, data, false);
}

template <typename T>
inline void Execute_IFFT_Pow2 (int32_t M, T* data)
{
    // In your real code: Same engine, but conjugate inputs/outputs or use inverse flag
    FFT_Pow2_Helper(M, data, true);
    
    // Normalize
    const T invM = static_cast<T>(1) / static_cast<T>(M);
    for (int i = 0; i < 2 * M; ++i) data[i] *= invM;
}

// ============================================================================
// PART 2: THE CHIRP-Z TRANSFORM (Bluestein's Algorithm)
// ============================================================================

template <typename T>
inline void fft_czt (const T* in, T* out, int32_t N)
{
    constexpr T PI = static_cast<T>(3.14159265358979323846);

    // 1. Determine Convolution Size M
    // The linear convolution of two sequences of length N requires size >= 2N - 1.
    // We choose the next power of 2 to allow ultra-fast FFTs.
    int32_t M = NextPowerOf2(2 * N - 1);

    // 2. Allocate Buffers (3 Complex Vectors)
    // In a production "Plan" class, these are allocated once and reused.
    // 'A' will hold the Pre-modulated Input
    // 'B' will hold the Chirp Kernel
    // 'W' will hold the Chirp values (reused for post-modulation)
    std::vector<T> A(2 * M, 0); 
    std::vector<T> B(2 * M, 0);
    std::vector<T> chirp(2 * N); // Only needs size N, but we store interleaved

    // -----------------------------------------------------------------------
    // 3. PRE-COMPUTE CHIRPS & FILL BUFFERS
    // -----------------------------------------------------------------------
    // The Chirp is W_k = exp( -j * PI * k^2 / N )
    // Note: The formula uses PI, not 2*PI, due to the n^2/2 identity.
    
    for (int k = 0; k < N; ++k)
	{
        // Calculate angle: -PI * k^2 / N
        T k_sq = static_cast<T>(1 * k * k); // 1LL prevents overflow
        T angle = -PI * k_sq / static_cast<T>(N);

        T c = static_cast<T>(std::cos(angle));
        T s = static_cast<T>(std::sin(angle));

        // Store Chirp for Post-Modulation
        chirp[2*k]     = c;
        chirp[2*k + 1] = s;

        // --- Fill Buffer A (Pre-Modulation) ---
        // A[k] = Input[k] * Chirp[k]
        T r_in = in[2*k];
        T i_in = in[2*k + 1];

        // Complex Mult: (r_in + j*i_in) * (c + j*s)
        A[2*k]     = r_in * c - i_in * s;
        A[2*k + 1] = r_in * s + i_in * c;
        
        // --- Fill Buffer B (Convolution Kernel) ---
        // The kernel represents W^(+(k-n)^2/2). 
        // This corresponds to Conjugate(Chirp).
        // It wraps around the buffer for circular convolution logic.
        
        // Positive Lags (0 to N-1)
        B[2*k]     = c;        // Real part of Conj(Chirp) = c
        B[2*k + 1] = -s;       // Imag part of Conj(Chirp) = -s
    }

    // Negative Lags (Wrap to end of M)
    // B[M-k] = B[k] for k = 1..N-1
    for (int32_t k = 1; k < N; ++k)
	{
        int32_t wrap_idx = M - k;
        B[2*wrap_idx]     = chirp[2*k];     // Real
        B[2*wrap_idx + 1] = -chirp[2*k+1];  // Imag (Conjugate)
    }

    // -----------------------------------------------------------------------
    // 4. PERFORM FAST CONVOLUTION
    // -----------------------------------------------------------------------
    
    // A. Forward FFT of Input (A) and Kernel (B)
    Execute_FFT_Pow2(M, A.data());
    Execute_FFT_Pow2(M, B.data());

    // B. Pointwise Multiplication in Frequency Domain
    // A = A * B
    for (int32_t k = 0; k < M; ++k)
	{
        T ar = A[2*k];     T ai = A[2*k + 1];
        T br = B[2*k];     T bi = B[2*k + 1];

        // Complex Mult
        A[2*k]     = ar * br - ai * bi;
        A[2*k + 1] = ar * bi + ai * br;
    }

    // C. Inverse FFT to get back to Time Domain
    Execute_IFFT_Pow2(M, A.data());

    // -----------------------------------------------------------------------
    // 5. POST-MODULATION & OUTPUT
    // -----------------------------------------------------------------------
    // The result of the convolution (A) is multiplied by the Chirp again.
    // Output[k] = A[k] * Chirp[k]
    // Only the first N elements are valid.

    for (int32_t k = 0; k < N; ++k)
	{
        T ar = A[2*k];
        T ai = A[2*k + 1];
        T c  = chirp[2*k];
        T s  = chirp[2*k + 1];

        // Complex Mult
        out[2*k]     = ar * c - ai * s;
        out[2*k + 1] = ar * s + ai * c;
    }
	
	return;
}

}