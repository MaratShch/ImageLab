#include <cmath>
#include "AVX2_AlgoMatrixMath.hpp"

// =========================================================
// AVX2 MATRIX MULTIPLICATION (A * B)
// =========================================================
void AVX2_Product_Matrix
(
    double* RESTRICT o_mat, 
    const double* RESTRICT i_A, 
    const double* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
) noexcept
{
    CACHE_ALIGN double q0[256];

    for (int32_t i = 0; i < p_l; i++) 
    {
        const double* pB = &i_B[i];
        for (int32_t k = 0; k < p_m; k++, pB += p_l) 
        {
            q0[k] = *pB;
        }

        double* pO = &o_mat[i];
        const double* pA = &i_A[0];

        for (int32_t j = 0; j < p_n; j++, pO += p_l, pA += p_m) 
        {
            __m256d vSum = _mm256_setzero_pd();
            int32_t k = 0;

            // Unroll by 4 doubles (256-bit AVX2)
            for (; k <= p_m - 4; k += 4) 
            {
                __m256d a = _mm256_loadu_pd(&pA[k]);
                __m256d b = _mm256_loadu_pd(&q0[k]);
                vSum = _mm256_fmadd_pd(a, b, vSum);
            }

            // Fast horizontal sum of the 4 doubles in the register
            __m128d vlow  = _mm256_castpd256_pd128(vSum);
            __m128d vhigh = _mm256_extractf128_pd(vSum, 1);
            vlow = _mm_add_pd(vlow, vhigh);
            __m128d shuf = _mm_unpackhi_pd(vlow, vlow);
            vlow = _mm_add_pd(vlow, shuf);
            double z = _mm_cvtsd_f64(vlow);

            // Tail loop for remaining elements
            for (; k < p_m; k++) 
            {
                z += pA[k] * q0[k];
            }

            *pO = z;
        }
    }
}

// =========================================================
// AVX2 COVARIANCE MATRIX (FLOAT INPUT)
// =========================================================
void AVX2_Covariance_Matrix
(
    const float* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
) noexcept
{
    const double coefNorm = 1.0 / static_cast<double>(p_nb - 1);

    for (int32_t i = 0; i < p_N; i++) 
    {
        const float* Pi = &i_patches[i * p_nb];
        double* cMi = &o_covMat[i * p_N];
        double* cMj = &o_covMat[i];

        for (int32_t j = 0; j < i + 1; j++) 
        {
            const float* Pj = &i_patches[j * p_nb];
            __m256d vSum = _mm256_setzero_pd();
            int32_t k = 0;

            for (; k <= p_nb - 4; k += 4) 
            {
                // Load 4 floats, convert directly to 4 doubles
                __m128 a_f = _mm_loadu_ps(&Pi[k]);
                __m128 b_f = _mm_loadu_ps(&Pj[k]);
                __m256d a_d = _mm256_cvtps_pd(a_f);
                __m256d b_d = _mm256_cvtps_pd(b_f);
                
                vSum = _mm256_fmadd_pd(a_d, b_d, vSum);
            }

            __m128d vlow  = _mm256_castpd256_pd128(vSum);
            __m128d vhigh = _mm256_extractf128_pd(vSum, 1);
            vlow = _mm_add_pd(vlow, vhigh);
            __m128d shuf = _mm_unpackhi_pd(vlow, vlow);
            vlow = _mm_add_pd(vlow, shuf);
            double val = _mm_cvtsd_f64(vlow);

            for (; k < p_nb; k++) 
            {
                val += static_cast<double>(Pi[k]) * static_cast<double>(Pj[k]);
            }

            cMj[j * p_N] = cMi[j] = val * coefNorm;
        }
    }
}

// =========================================================
// AVX2 COVARIANCE MATRIX (DOUBLE INPUT)
// =========================================================
void AVX2_Covariance_Matrix
(
    const double* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
) noexcept
{
    const double coefNorm = 1.0 / static_cast<double>(p_nb - 1);

    for (int32_t i = 0; i < p_N; i++) 
    {
        const double* Pi = &i_patches[i * p_nb];
        double* cMi = &o_covMat[i * p_N];
        double* cMj = &o_covMat[i];

        for (int32_t j = 0; j < i + 1; j++) 
        {
            const double* Pj = &i_patches[j * p_nb];
            __m256d vSum = _mm256_setzero_pd();
            int32_t k = 0;

            for (; k <= p_nb - 4; k += 4) 
            {
                __m256d a = _mm256_loadu_pd(&Pi[k]);
                __m256d b = _mm256_loadu_pd(&Pj[k]);
                vSum = _mm256_fmadd_pd(a, b, vSum);
            }

            __m128d vlow  = _mm256_castpd256_pd128(vSum);
            __m128d vhigh = _mm256_extractf128_pd(vSum, 1);
            vlow = _mm_add_pd(vlow, vhigh);
            __m128d shuf = _mm_unpackhi_pd(vlow, vlow);
            vlow = _mm_add_pd(vlow, shuf);
            double val = _mm_cvtsd_f64(vlow);

            for (; k < p_nb; k++) 
            {
                val += Pi[k] * Pj[k];
            }

            cMj[j * p_N] = cMi[j] = val * coefNorm;
        }
    }
}

// =========================================================
// SCALAR FALLBACKS WITH AVX2 PREFIX (DATA DEPENDENCY)
// =========================================================
void AVX2_Transpose_Matrix(double* RESTRICT io_mat, const int32_t p_N) noexcept
{
    for (int32_t i = 0; i < p_N - 1; i++)
	{
        int32_t p = i * (p_N + 1) + 1;
        int32_t q = i * (p_N + 1) + p_N;
        for (int32_t j = 0; j < p_N - 1 - i; j++, p++, q += p_N)
		{
            const double s = io_mat[p];
            io_mat[p] = io_mat[q];
            io_mat[q] = s;
        }
    }
}

bool AVX2_Inverse_Matrix (double* RESTRICT io_mat, const int32_t p_N) noexcept
{
    double z;
    int32_t p, q, r, s, t, j, k;

    for (j = 0, p = 0; j < p_N ; j++, p += p_N + 1)
	{
        for (q = j * p_N; q < p ; q++) io_mat[p] -= io_mat[q] * io_mat[q];
        if (io_mat[p] <= 0.05) return false; 
        io_mat[p] = std::sqrt(io_mat[p]);
        for (k = j + 1, q = p + p_N; k < p_N ; k++, q += p_N)
		{
            for (r = j * p_N, s = k * p_N, z = 0; r < p; r++, s++) z += io_mat[r] * io_mat[s];
            io_mat[q] -= z;
            io_mat[q] /= io_mat[p];
        }
    }
    AVX2_Transpose_Matrix(io_mat, p_N);
    for (j = 0, p = 0; j < p_N; j++, p += p_N + 1)
	{
        io_mat[p] = 1.0 / io_mat[p];
        for (q = j, t = 0; q < p; t += p_N + 1, q += p_N)
		{
            for (s = q, r = t, z = 0; s < p; s += p_N, r++) z -= io_mat[s] * io_mat[r];
            io_mat[q] = z * io_mat[p];
        }
    }
    for (j = 0, p = 0; j < p_N; j++, p += p_N + 1)
	{
        for (q = j, t = p - j; q <= p; q += p_N, t++)
		{
            for (k = j, r = p, s = q, z = 0; k < p_N; k++, r++, s++) 
				z += io_mat[r] * io_mat[s];
            io_mat[t] = io_mat[q] = z;
        }
    }
    return true;
}