#include <cmath>
#include <immintrin.h>
#include "AVX2_Smpl_AlgoMatrixMath.hpp"

// =========================================================
// FAST AVX2 DOT PRODUCT HELPER
// =========================================================
inline float AVX2_DotProduct_PS(const float* a, const float* b, const int32_t count) noexcept 
{
    __m256 vSum = _mm256_setzero_ps();
    int32_t i = 0;
    
    // Process 8 floats simultaneously
    for (; i <= count - 8; i += 8) 
    {
        vSum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vSum);
    }
    
    // Fast horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vSum);
    __m128 vhigh = _mm256_extractf128_ps(vSum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
    vlow = _mm_add_ss(vlow, shuf);
    float sum = _mm_cvtss_f32(vlow);
    
    // Scalar tail
    for (; i < count; ++i) 
    {
        sum += a[i] * b[i];
    }
    return sum;
}

// =========================================================
// SYMMETRIC REAL MATRIX INVERSION (AVX2 ACCELERATED)
// =========================================================
bool AVX2_Smpl_Inverse_Matrix (float* RESTRICT io_mat, const int32_t p_N)
{
    float z;
    int32_t p, q, r, s, t, j, k;

    // Phase 1: Cholesky-style decomposition
    for (j = 0, p = 0; j < p_N ; j++, p += p_N + 1) 
    {
        int32_t count = p - (j * p_N);
        if (count > 0) 
        {
            io_mat[p] -= AVX2_DotProduct_PS(&io_mat[j * p_N], &io_mat[j * p_N], count);
        }

        if (io_mat[p] <= 0.0001f) return false; 
        io_mat[p] = std::sqrt(io_mat[p]);

        for (k = j + 1, q = p + p_N; k < p_N ; k++, q += p_N) 
        {
            if (count > 0)
            {
                z = AVX2_DotProduct_PS(&io_mat[j * p_N], &io_mat[k * p_N], count);
                io_mat[q] -= z;
            }
            io_mat[q] /= io_mat[p];
        }
    }

    AVX2_Smpl_Transpose_Matrix(io_mat, p_N);

    // Phase 2: Invert the diagonal matrix
    for (j = 0, p = 0; j < p_N; j++, p += p_N + 1) 
    {
        io_mat[p] = 1.0f / io_mat[p];

        for (q = j, t = 0; q < p; t += p_N + 1, q += p_N) 
        {
            for (s = q, r = t, z = 0.0f; s < p; s += p_N, r++) 
            {
                z -= io_mat[s] * io_mat[r];
            }
            io_mat[q] = z * io_mat[p];
        }
    }

    // Phase 3: Recombine
    for (j = 0, p = 0; j < p_N; j++, p += p_N + 1) 
    {
        for (q = j, t = p - j; q <= p; q += p_N, t++) 
        {
            int32_t tail_count = p_N - j;
            z = AVX2_DotProduct_PS(&io_mat[p], &io_mat[q], tail_count);
            io_mat[t] = io_mat[q] = z;
        }
    }

    return true;
}

// =========================================================
// MATRIX TRANSPOSE (IN-PLACE)
// =========================================================
void AVX2_Smpl_Transpose_Matrix (float* RESTRICT io_mat, const int32_t p_N)
{
    for (int32_t i = 0; i < p_N - 1; i++) 
    {
        int32_t p = i * (p_N + 1) + 1;
        int32_t q = i * (p_N + 1) + p_N;

        for (int32_t j = 0; j < p_N - 1 - i; j++, p++, q += p_N) 
        {
            const float s = io_mat[p];
            io_mat[p] = io_mat[q];
            io_mat[q] = s;
        }
    }
}

// =========================================================
// MATRIX MULTIPLICATION (AVX2 CACHE-FRIENDLY BROADCAST)
// =========================================================
void AVX2_Smpl_Product_Matrix
(
    float* RESTRICT o_mat, 
    const float* RESTRICT i_A, 
    const float* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
)
{
    // i_A is [p_n x p_m], i_B is [p_m x p_l], o_mat is [p_n x p_l]
    for (int32_t i = 0; i < p_n; ++i) 
    {
        // Zero out the output row
        for (int32_t j = 0; j < p_l; ++j) o_mat[i * p_l + j] = 0.0f;
        
        for (int32_t k = 0; k < p_m; ++k) 
        {
            // Broadcast a single value from matrix A
            float a_val = i_A[i * p_m + k];
            __m256 vA = _mm256_set1_ps(a_val);
            
            int32_t j = 0;
            // Multiply-Add directly into the output row using matrix B
            for (; j <= p_l - 8; j += 8) 
            {
                __m256 vB = _mm256_loadu_ps(&i_B[k * p_l + j]);
                __m256 vO = _mm256_loadu_ps(&o_mat[i * p_l + j]);
                _mm256_storeu_ps(&o_mat[i * p_l + j], _mm256_fmadd_ps(vA, vB, vO));
            }
            
            // Scalar tail (only executes if p_l is not divisible by 8)
            for (; j < p_l; ++j) 
            {
                o_mat[i * p_l + j] += a_val * i_B[k * p_l + j];
            }
        }
    }
}