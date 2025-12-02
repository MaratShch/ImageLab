#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>
#include <cstring>
#include "Common.hpp"
#include "FastAriphmetics.hpp"

namespace FourierTransform
{
    // =========================================================
    // PROLEVEL CONSTANTS & TYPES
    // =========================================================

    // Internal helper for the manual stack
    struct Task
    {
        size_t offset;
        size_t n;
    };

    // =========================================================
    // HELPER: Interleaved Bit-Reversal
    // Swaps (Re, Im) pairs to unscramble the DIF output
    // =========================================================
    template <typename T>
    inline void bit_reverse_interleaved(T* data, const size_t n)
    {
        size_t j = 0;
        for (size_t i = 0; i < n; ++i)
        {
            if (j > i)
            {
                // Swap pair (Re, Im) at index i with pair at index j
                // Data is interleaved, so index i starts at 2*i
                size_t idx_i = i << 1; // i * 2
                size_t idx_j = j << 1; // j * 2

                std::swap(data[idx_i], data[idx_j]);     // Re
                std::swap(data[idx_i + 1], data[idx_j + 1]); // Im
            }
            size_t m = n >> 1;
            while (m >= 1 && j >= m)
            {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
    }

    // =========================================================
    // CORE ALGORITHM: Iterative Split-Radix (Interleaved)
    // =========================================================
    template <typename T>
    inline void fft_split_radix_interleaved(const T* input, T* output, size_t n)
    {
        constexpr double PI = 3.14159265358979323846;

        // 1. Validation
        if (n == 0 || (n & (n - 1)) != 0) return; // Must be power of 2

        // 2. Out-of-Place: Copy Input to Output first
        // We process 'output' in-place from here on.
        // 2*n because we have Re+Im for every sample.
        for (size_t i = 0; i < 2 * n; ++i)
        {
            output[i] = input[i];
        }

        // 3. Create Explicit Stack (on Heap to prevent stack overflow)
        // Reserve memory: log2(N) + small buffer is sufficient.
        std::vector<Task> stack(64);

        // Push initial job: Process the whole array starting at 0
        stack.push_back({ 0, n });

        // 4. Main Iterative Loop
        while (!stack.empty())
        {
            Task task = stack.back();
            stack.pop_back();

            size_t task_n = task.n;
            size_t offset = task.offset; // Offset in "samples" (not doubles)

            // --- Base Case: Leaf ---
            if (task_n == 1) continue;

            // --- Base Case: Radix-2 Butterfly (N=2) ---
            if (task_n == 2)
            {
                // Interleaved indices:
                // Point 0: 2*offset,     2*offset+1
                // Point 1: 2*(offset+1), 2*(offset+1)+1

                size_t i0 = offset << 1;
                size_t i1 = (offset + 1) << 1;

                T r0 = output[i0];     T i0_val = output[i0 + 1];
                T r1 = output[i1];     T i1_val = output[i1 + 1];

                // Butterfly
                output[i0] = r0 + r1;
                output[i0 + 1] = i0_val + i1_val;
                output[i1] = r0 - r1;
                output[i1 + 1] = i0_val - i1_val;
                continue;
            }

            // --- Split-Radix Decomposition ---
            size_t n_2 = task_n >> 1; // N/2
            size_t n_4 = task_n >> 2; // N/4

            // We iterate k from 0 to N/4 - 1
            for (size_t k = 0; k < n_4; ++k)
            {
                // Calculate indices for the 4 points in the L-shape
                // Standard Split-Radix DIF indices:
                // x0: k
                // x1: k + N/4
                // x2: k + N/2
                // x3: k + 3N/4

                // Convert to Interleaved Memory offsets (multiply by 2)
                size_t idx0 = (offset + k) << 1;
                size_t idx1 = (offset + k + n_4) << 1;
                size_t idx2 = (offset + k + n_2) << 1;
                size_t idx3 = (offset + k + 3 * n_4) << 1;

                // --- Load Data ---
                T r0 = output[idx0]; T i0 = output[idx0 + 1];
                T r1 = output[idx1]; T i1 = output[idx1 + 1];
                T r2 = output[idx2]; T i2 = output[idx2 + 1];
                T r3 = output[idx3]; T i3 = output[idx3 + 1];

                // --- Even Part Sums (Standard Radix-2) ---
                // U0 = x0 + x2
                T u0_r = r0 + r2;
                T u0_i = i0 + i2;
                // U1 = x1 + x3
                T u1_r = r1 + r3;
                T u1_i = i1 + i3;

                // --- Odd Part Diffs ---
                // D0 = x0 - x2
                T d0_r = r0 - r2;
                T d0_i = i0 - i2;
                // D1 = x1 - x3
                T d1_r = r1 - r3;
                T d1_i = i1 - i3;

                // --- Rotation by -j ---
                // T0 = D0 - j*D1 = (d0_r + d1_i) + j(d0_i - d1_r)
                T t0_r = d0_r + d1_i;
                T t0_i = d0_i - d1_r;

                // T1 = D0 + j*D1 = (d0_r - d1_i) + j(d0_i + d1_r)
                T t1_r = d0_r - d1_i;
                T t1_i = d0_i + d1_r;

                // --- Twiddle Factors ---
                // Angle = -2*PI*k / N
                // Note: Use task_n (current block size)
                T angle = static_cast<T>(-2.0 * PI) * static_cast<T>(k) / static_cast<T>(task_n);

                T c1 = std::cos(angle);
                T s1 = std::sin(angle);
                T c3 = std::cos(static_cast<T>(3) * angle);
                T s3 = std::sin(static_cast<T>(3) * angle);

                // --- Store Even Parts ---
                // Stored in the lower half (indices idx0 and idx1)
                output[idx0] = u0_r + u1_r;
                output[idx0 + 1] = u0_i + u1_i;

                output[idx1] = u0_r - u1_r;
                output[idx1 + 1] = u0_i - u1_i;

                // --- Apply Twiddles & Store Odd Parts ---
                // Stored in the upper half (indices idx2 and idx3)

                // x[idx2] = T0 * W^k
                output[idx2] = t0_r * c1 - t0_i * s1;
                output[idx2 + 1] = t0_r * s1 + t0_i * c1;

                // x[idx3] = T1 * W^3k
                output[idx3] = t1_r * c3 - t1_i * s3;
                output[idx3 + 1] = t1_r * s3 + t1_i * c3;
            }

            // --- Push Sub-Tasks (Reverse Order) ---
            // 1. Odd Part 2 (Top 1/4 of memory)
            stack.push_back({ offset + 3 * n_4, n_4 });
            // 2. Odd Part 1 (Next 1/4 of memory)
            stack.push_back({ offset + n_2, n_4 });
            // 3. Even Part (Lower 1/2 of memory) - Process THIS next
            stack.push_back({ offset, n_2 });
        }

        // 5. Unscramble the results
        bit_reverse_interleaved(output, n);
    }

}