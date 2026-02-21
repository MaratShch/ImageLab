#pragma once

#include <cstdint>
#include <algorithm>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"

void Estimate_Noise_Covariances
(
    const MemHandler& mem,
    const int32_t width,
    const int32_t height,
    const AlgoControls& algoCtrl
);

// =========================================================
// DATA STRUCTURES & CONSTANTS
// =========================================================
struct OracleCandidate 
{
    int16_t x, y;
    float mean;
    float sd;
    float dct[16];
};

struct BinInfo 
{
    int32_t start_idx;
    int32_t count;
    float mean_val;
    float stds[16];
};

// =========================================================
// INLINE MATH & ORACLE HELPERS
// =========================================================

// Checks if a 2x2 area is completely flat or clipped (artificial edge)
inline bool Is_Valid_Block (const float* RESTRICT img, int32_t x, int32_t y, int32_t stride) noexcept 
{
    const float p0 = img[y * stride + x];
    const float p1 = img[y * stride + x + 1];
    const float p2 = img[(y + 1) * stride + x];
    const float p3 = img[(y + 1) * stride + x + 1];
    
    if (std::abs(p0 - p1) < 0.001f && std::abs(p1 - p2) < 0.001f && std::abs(p2 - p3) < 0.001f) {
        return false; 
    }
    return true;
}

// Precomputes the Orthonormal 2D DCT Basis matrix D[16][16]
inline void Generate_DCT_Basis(float D[16][16]) noexcept 
{
    constexpr float pi = 3.14159265358979323846f;
    for (int32_t u = 0; u < 4; ++u)
    {
        for (int32_t v = 0; v < 4; ++v)
        {
            const int32_t k = u * 4 + v;
            const float alpha_u = (u == 0) ? 0.5f : 0.707106781f;
            const float alpha_v = (v == 0) ? 0.5f : 0.707106781f;
            
            for (int32_t y = 0; y < 4; ++y)
            {
                for (int32_t x = 0; x < 4; ++x)
                {
                    const int32_t p = y * 4 + x;
                    D[k][p] = alpha_u * alpha_v * std::cos(pi * u * (2.0f * x + 1.0f) / 8.0f) * std::cos(pi * v * (2.0f * y + 1.0f) / 8.0f);
                }
            }
        }
    }
}

// Replaces FFTW: Projects a 4x4 spatial block into DCT frequencies
inline void Forward_DCT_4x4
(
    const float* RESTRICT block, 
    const int32_t stride, 
    float* RESTRICT dct, 
    const float D[16][16]
) noexcept 
{
    for (int32_t k = 0; k < 16; ++k)
    {
        float sum = 0.0f;
        for (int32_t y = 0; y < 4; ++y)
        {
            for (int32_t x = 0; x < 4; ++x)
            {
                sum += D[k][y * 4 + x] * block[y * stride + x];
            }
        }
        dct[k] = sum;
    }
}

// Matches Colom's 'compute_sparse_distance'
inline float Calculate_Sparse_Distance (const float* RESTRICT dctA, const float* RESTRICT dctB) noexcept 
{
    struct Diff { float abs_diff; float penalty; };
    CACHE_ALIGN Diff diffs[16];
    
    for(int32_t i = 0; i < 16; ++i)
    {
        float a = dctA[i];
        float b = dctB[i];
        diffs[i].abs_diff = std::abs(a - b);
        diffs[i].penalty = std::abs(a - b) * std::max(std::abs(a), std::abs(b));
    }
    
    // Insertion sort to find the smallest 12 differences (ignores the 4 largest)
    for (int32_t i = 1; i < 16; ++i)
    {
        Diff key = diffs[i];
        int32_t j = i - 1;
        while (j >= 0 && diffs[j].abs_diff > key.abs_diff)
        {
            diffs[j + 1] = diffs[j];
            j = j - 1;
        }
        diffs[j + 1] = key;
    }
    
    float sd = 0.0f;
    for (int32_t i = 0; i < 12; ++i)
    { 
        sd += diffs[i].penalty;
    }
    return sd;
}

// Computes Median Absolute Deviation
inline float Compute_MAD(float* RESTRICT arr, const int32_t size) noexcept 
{
    if (size == 0) return 0.0f;
    std::sort(arr, arr + size);
    float median = (size % 2 == 0) ? (arr[size/2 - 1] + arr[size/2]) * 0.5f : arr[size/2];
    
    for (int32_t i = 0; i < size; ++i) {
        arr[i] = std::abs(arr[i] - median);
    }
    
    std::sort(arr, arr + size);
    return (size % 2 == 0) ? (arr[size/2 - 1] + arr[size/2]) * 0.5f : arr[size/2];
}