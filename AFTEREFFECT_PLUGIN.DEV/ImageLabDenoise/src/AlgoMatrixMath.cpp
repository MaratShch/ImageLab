#include <cmath>
#include "AlgoMatrixMath.hpp"

// =========================================================
// SYMMETRIC REAL MATRIX INVERSION
// Original Author: Daniel A. Atkinson (ccmath)
// =========================================================
bool Inverse_Matrix (double* RESTRICT io_mat, const int32_t p_N)
{
    double z;
    int32_t p, q, r, s, t, j, k;

    // Phase 1: Cholesky-style decomposition
    for (j = 0, p = 0; j < p_N ; j++, p += p_N + 1) 
    {
        for (q = j * p_N; q < p ; q++) 
        {
            io_mat[p] -= io_mat[q] * io_mat[q];
        }

        // Failsafe: Matrix is not positive definite
        if (io_mat[p] <= 0.05) 
        {
            return false; 
        }

        io_mat[p] = std::sqrt(io_mat[p]);

        for (k = j + 1, q = p + p_N; k < p_N ; k++, q += p_N) 
        {
            for (r = j * p_N, s = k * p_N, z = 0; r < p; r++, s++) 
            {
                z += io_mat[r] * io_mat[s];
            }

            io_mat[q] -= z;
            io_mat[q] /= io_mat[p];
        }
    }

    Transpose_Matrix(io_mat, p_N);

    // Phase 2: Invert the diagonal matrix
    for (j = 0, p = 0; j < p_N; j++, p += p_N + 1) 
    {
        io_mat[p] = 1.0 / io_mat[p];

        for (q = j, t = 0; q < p; t += p_N + 1, q += p_N) 
        {
            for (s = q, r = t, z = 0; s < p; s += p_N, r++) 
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
            for (k = j, r = p, s = q, z = 0; k < p_N; k++, r++, s++) 
            {
                z += io_mat[r] * io_mat[s];
            }

            io_mat[t] = io_mat[q] = z;
        }
    }

    return true;
}

// =========================================================
// MATRIX TRANSPOSE (IN-PLACE)
// =========================================================
void Transpose_Matrix (double* RESTRICT io_mat, const int32_t p_N)
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

// =========================================================
// COVARIANCE MATRIX (FLOAT ARRAY INPUT)
// =========================================================
void Covariance_Matrix
(
    const float* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
)
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
            double val = 0.0;

            for (int32_t k = 0; k < p_nb; k++) 
            {
                val += static_cast<double>(Pi[k]) * static_cast<double>(Pj[k]);
            }

            cMj[j * p_N] = cMi[j] = val * coefNorm;
        }
    }
}

// =========================================================
// COVARIANCE MATRIX (DOUBLE ARRAY INPUT)
// =========================================================
void Covariance_Matrix
(
    const double* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
)
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
            double val = 0.0;

            for (int32_t k = 0; k < p_nb; k++) 
            {
                val += Pi[k] * Pj[k];
            }

            cMj[j * p_N] = cMi[j] = val * coefNorm;
        }
    }
}

// =========================================================
// MATRIX MULTIPLICATION (A * B)
// =========================================================
void Product_Matrix
(
    double* RESTRICT o_mat, 
    const double* RESTRICT i_A, 
    const double* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
)
{
    // Replaces dynamic std::vector<double> q0(p_m).
    // The max patch size in NC is 4x4 (p_m = 16), so 256 is extremely safe.
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
            double z = 0.0;

            for (int32_t k = 0; k < p_m; k++) 
            {
                z += pA[k] * q0[k];
            }

            *pO = z;
        }
    }
}