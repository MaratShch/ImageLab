#include <cstdint>
#include "MosaicMemHandler.hpp"
#include "ImageMosaicUtils.hpp"
#include "FastAriphmetics.hpp"

// #define _ALGO_PLANAR_BUFFERS_CHECK_

inline float planarSqNormGradient
(
    const float* RESTRICT R,
    const float* RESTRICT G,
    const float* RESTRICT B, 
    A_long cx,
    A_long cy,
    A_long width,
    A_long height
) noexcept
{
    float gradient = 0.0f;
    const A_long idx = cy * width + cx;

    // Right neighbor (X + 1)
    A_long nx = cx + 1;
    if (nx >= width) nx = cx - 1;
    if (nx < 0) nx = cx;
    const A_long idxX = cy * width + nx;

    float dr = R[idxX] - R[idx];
    float dg = G[idxX] - G[idx];
    float db = B[idxX] - B[idx];
    gradient += (dr * dr + dg * dg + db * db);

    // Bottom neighbor (Y + 1)
    A_long ny = cy + 1;
    if (ny >= height) ny = cy - 1;
    if (ny < 0) ny = cy;
    const A_long idxY = ny * width + cx;

    dr = R[idxY] - R[idx];
    dg = G[idxY] - G[idx];
    db = B[idxY] - B[idx];
    gradient += (dr * dr + dg * dg + db * db);

    return gradient;
}

void MosaicAlgorithmMain (const MemHandler& memHndl, A_long width, A_long height, A_long K)
{
    constexpr float m = 40.f;
    constexpr int maxNorm = std::numeric_limits<unsigned char>::max();
    constexpr ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
    constexpr ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

    A_long k = K;

    // --- 1. INITIALIZATION ---
    
    const A_long numPixels = width * height;
    // Calculate grid interval S
    const float superPixInitVal = static_cast<float>(numPixels) / static_cast<float>(K);
    const A_long S = FastCompute::Max(1, static_cast<A_long>(FastCompute::Sqrt(superPixInitVal)));

    const A_long nX = FastCompute::Max(1, width / S);
    const A_long nY = FastCompute::Max(1, height / S);

    const A_long padw = FastCompute::Max(0, width - S * nX);
    const A_long padh = FastCompute::Max(0, height - S * nY);
    
    const A_long s_half = S >> 1;
    const A_long halfPadW = padw >> 1;
    const A_long halfPadH = padh >> 1;

    A_long actualK = 0;

    // Distribute centers across the grid
    for (A_long j = 0; j < nY; j++)
    {
        const A_long jj = j * S + s_half + halfPadH;
        for (A_long i = 0; i < nX; i++)
        {
            const A_long ii = i * S + s_half + halfPadW;
            
            if (ii < width && jj < height)
            {
                // In our planar buffers, the index is strictly contiguous based on width
                const A_long planarIdx = jj * width + ii; 

                // Populate the SoA superpixel buffers directly
                memHndl.sp_X[actualK] = static_cast<float>(ii);
                memHndl.sp_Y[actualK] = static_cast<float>(jj);
                memHndl.sp_R[actualK] = memHndl.R_planar[planarIdx];
                memHndl.sp_G[actualK] = memHndl.G_planar[planarIdx];
                memHndl.sp_B[actualK] = memHndl.B_planar[planarIdx];
                
                actualK++;
            }
        }
    }

    // Update K to reflect the exact number of grid points generated
    k = actualK; 

    // --- 1.5 GRADIENT PERTURBATION (Stage 1) ---
    // Nudge centers to the lowest gradient position in a 3x3 window
    const A_long searchRadius = 1; // 1 pixel radius = 3x3 window

    for (A_long k_idx = 0; k_idx < k; k_idx++)
    {
        const A_long startX = static_cast<A_long>(memHndl.sp_X[k_idx]);
        const A_long startY = static_cast<A_long>(memHndl.sp_Y[k_idx]);
        
        float minGradient = std::numeric_limits<float>::max();
        A_long bestX = startX;
        A_long bestY = startY;

        // Scan the local neighborhood
        for (A_long j = -searchRadius; j <= searchRadius; j++)
        {
            for (A_long i = -searchRadius; i <= searchRadius; i++)
            {
                A_long currX = startX + i;
                A_long currY = startY + j;

                if (currX >= 0 && currX < width && currY >= 0 && currY < height)
                {
                    float g = planarSqNormGradient(
                        memHndl.R_planar, memHndl.G_planar, memHndl.B_planar, 
                        currX, currY, width, height
                    );
                    
                    if (g < minGradient)
                    {
                        minGradient = g;
                        bestX = currX;
                        bestY = currY;
                    }
                }
            }
        }

        // Update the SoA superpixel buffers to the newly found best position
        const A_long bestIdx = bestY * width + bestX;
        memHndl.sp_X[k_idx] = static_cast<float>(bestX);
        memHndl.sp_Y[k_idx] = static_cast<float>(bestY);
        memHndl.sp_R[k_idx] = memHndl.R_planar[bestIdx];
        memHndl.sp_G[k_idx] = memHndl.G_planar[bestIdx];
        memHndl.sp_B[k_idx] = memHndl.B_planar[bestIdx];
    }
    
#ifdef _ALGO_PLANAR_BUFFERS_CHECK_
    std::cout<<std::endl;
    CheckPlanarRange(memHndl.R_planar, numPixels, "Red    Input");
    CheckPlanarRange(memHndl.G_planar, numPixels, "Green Input");
    CheckPlanarRange(memHndl.B_planar, numPixels, "Blue   Input");
#endif // _ALGO_PLANAR_BUFFERS_CHECK_

    // --- 2. MAIN SLIC ITERATIONS ---
    
    constexpr A_long maxIter = 10;
    const float wSpace = m / static_cast<float>(S);
    const float wSpaceSq = wSpace * wSpace; // Pre-calculate the squared weight!
    
    // Vectorized constants for the inner loop
    const __m256 v_wSpaceSq = _mm256_set1_ps(wSpaceSq);
    const __m256 v_seq_8 = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

    for (A_long iter = 0; iter < maxIter; iter++)
    {
        // 1. Reset Distance buffer to Float Max
        ArtMosaic::fillProcBuf(memHndl.D, numPixels, std::numeric_limits<float>::max());
        ArtMosaic::fillProcBuf(memHndl.L, numPixels, -1);
        
        // 2. Assignment Step: Loop over every superpixel center
        for (A_long k_idx = 0; k_idx < k; k_idx++)
        {
            // Load the center's SoA data
            const float cX = memHndl.sp_X[k_idx];
            const float cY = memHndl.sp_Y[k_idx];
            const float cR = memHndl.sp_R[k_idx];
            const float cG = memHndl.sp_G[k_idx];
            const float cB = memHndl.sp_B[k_idx];

            // Broadcast center data to AVX registers
            const __m256 v_cX = _mm256_set1_ps(cX);
            const __m256 v_cY = _mm256_set1_ps(cY);
            const __m256 v_cR = _mm256_set1_ps(cR);
            const __m256 v_cG = _mm256_set1_ps(cG);
            const __m256 v_cB = _mm256_set1_ps(cB);
            const __m256i v_k_idx = _mm256_set1_epi32(k_idx);

            // Calculate exact bounding box (2S x 2S window)
            const A_long minX = FastCompute::Max(0L, static_cast<A_long>(cX + 0.5f) - S);
            const A_long maxX = FastCompute::Min(width, static_cast<A_long>(cX + 0.5f) + S);
            const A_long minY = FastCompute::Max(0L, static_cast<A_long>(cY + 0.5f) - S);
            const A_long maxY = FastCompute::Min(height, static_cast<A_long>(cY + 0.5f) + S);

            // 3. The Spatial Loop
            for (A_long y = minY; y < maxY; y++)
            {
                const __m256 v_y = _mm256_set1_ps(static_cast<float>(y));
                const A_long rowOffset = y * width;
                
                // Calculate how many pixels we can process in chunks of 8
                const A_long spanX = maxX - minX;
                const A_long spanX8 = spanX & ~7; // Equivalent to (spanX / 8) * 8
                
                A_long x = minX;
                
                // --- THE AVX2 INNER LOOP GOES HERE ---
                for (; x < minX + spanX8; x += 8)
                {
                    const A_long idx = rowOffset + x;

                    // 1. Calculate the 8 spatial X coordinates
                    const __m256 v_x = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), v_seq_8);

                    // 2. Spatial Distance: (dx^2 + dy^2) * wSpaceSq
                    const __m256 v_dx = _mm256_sub_ps(v_x, v_cX);
                    const __m256 v_dy = _mm256_sub_ps(v_y, v_cY);
                    __m256 v_distSq = _mm256_fmadd_ps(v_dx, v_dx, _mm256_mul_ps(v_dy, v_dy));
                    v_distSq = _mm256_mul_ps(v_distSq, v_wSpaceSq); 

                    // --- EARLY EXIT OPTIMIZATION ---
                    // Load the existing minimum distance right now
                    __m256 v_oldDist = _mm256_loadu_ps(&memHndl.D[idx]);
                    
                    // Check if spatial distance ALONE is already greater than the existing total distance
                    __m256 v_early_mask = _mm256_cmp_ps(v_distSq, v_oldDist, _CMP_LT_OQ); 
                    
                    // If NO pixels in this 8-pack have a smaller spatial distance, 
                    // color distance cannot possibly help. Skip memory loads!
                    if (_mm256_testz_ps(v_early_mask, v_early_mask))
                    {
                        continue;
                    }
                    // ------------------------------

                    // 3. Load Color (Direct planar loads)
                    const __m256 v_r = _mm256_loadu_ps(&memHndl.R_planar[idx]);
                    const __m256 v_g = _mm256_loadu_ps(&memHndl.G_planar[idx]);
                    const __m256 v_b = _mm256_loadu_ps(&memHndl.B_planar[idx]);

                    // 4. Color Distance: dr^2 + dg^2 + db^2
                    const __m256 v_dr = _mm256_sub_ps(v_r, v_cR);
                    const __m256 v_dg = _mm256_sub_ps(v_g, v_cG);
                    const __m256 v_db = _mm256_sub_ps(v_b, v_cB);

                    __m256 v_colorSq = _mm256_fmadd_ps(v_dr, v_dr, _mm256_mul_ps(v_dg, v_dg));
                    v_colorSq = _mm256_fmadd_ps(v_db, v_db, v_colorSq);

                    // 5. Total SLIC Distance
                    __m256 v_TotalDist = _mm256_add_ps(v_distSq, v_colorSq);

                    // 6. Compare with existing minimum distance (v_oldDist already loaded)
                    __m256 v_mask = _mm256_cmp_ps(v_TotalDist, v_oldDist, _CMP_LT_OQ); 

                    // 7. Masked Update for Distance Buffer
                    __m256 v_newDist = _mm256_blendv_ps(v_oldDist, v_TotalDist, v_mask);
                    _mm256_storeu_ps(&memHndl.D[idx], v_newDist);

                    // 8. Masked Update for Label Buffer
                    __m256i v_oldL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&memHndl.L[idx]));
                    __m256i v_newL = _mm256_blendv_epi8(v_oldL, v_k_idx, _mm256_castps_si256(v_mask));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&memHndl.L[idx]), v_newL);              
                }

                // --- SCALAR TAIL LOOP ---
                for (; x < maxX; x++)
                {
                    const A_long idx = rowOffset + x;

                    const float dx = static_cast<float>(x) - cX;
                    const float dy = static_cast<float>(y) - cY;
                    const float distSq = (dx * dx + dy * dy) * wSpaceSq;

                    if (distSq >= memHndl.D[idx]) continue; // Scalar Early Exit

                    const float r = memHndl.R_planar[idx];
                    const float g = memHndl.G_planar[idx];
                    const float b = memHndl.B_planar[idx];

                    const float dr = r - cR;
                    const float dg = g - cG;
                    const float db = b - cB;
                    const float colorSq = (dr * dr) + (dg * dg) + (db * db);

                    const float totalDist = distSq + colorSq;

                    if (totalDist < memHndl.D[idx])
                    {
                        memHndl.D[idx] = totalDist;
                        memHndl.L[idx] = k_idx;
                    }
                }               
            }
        } // End of Assignment Step
        
#ifdef _ALGO_PLANAR_BUFFERS_CHECK_
        std::cout << "Iteration: " << iter << std::endl;
        CheckPlanarRange(memHndl.L, numPixels, "L output");
        CheckPlanarRange(memHndl.D, numPixels, "D output");
#endif // _ALGO_PLANAR_BUFFERS_CHECK_

        // --- 3. UPDATE STEP (ZERO ALLOCATION) ---
        
        ArtMosaic::fillProcBuf(memHndl.acc_X, k, 0.0f);
        ArtMosaic::fillProcBuf(memHndl.acc_Y, k, 0.0f);
        ArtMosaic::fillProcBuf(memHndl.acc_R, k, 0.0f);
        ArtMosaic::fillProcBuf(memHndl.acc_G, k, 0.0f);
        ArtMosaic::fillProcBuf(memHndl.acc_B, k, 0.0f);
        ArtMosaic::fillProcBuf(memHndl.acc_Count, k, 0);

        for (A_long j = 0; j < height; j++) 
        {
            const A_long rowOffset = j * width;
            for (A_long i = 0; i < width; i++) 
            {
                const A_long idx = rowOffset + i;
                const A_long label = memHndl.L[idx];

                if (label >= 0) 
                {
                    memHndl.acc_X[label] += static_cast<float>(i);
                    memHndl.acc_Y[label] += static_cast<float>(j);
                    memHndl.acc_R[label] += memHndl.R_planar[idx];
                    memHndl.acc_G[label] += memHndl.G_planar[idx];
                    memHndl.acc_B[label] += memHndl.B_planar[idx];
                    memHndl.acc_Count[label]++;
                }
            }
        }

        for (A_long k_idx = 0; k_idx < k; k_idx++) 
        {
            const int32_t count = memHndl.acc_Count[k_idx];
            if (count > 0) 
            {
                const float invCount = 1.0f / static_cast<float>(count);
                
                memHndl.sp_X[k_idx] = memHndl.acc_X[k_idx] * invCount;
                memHndl.sp_Y[k_idx] = memHndl.acc_Y[k_idx] * invCount;
                memHndl.sp_R[k_idx] = memHndl.acc_R[k_idx] * invCount;
                memHndl.sp_G[k_idx] = memHndl.acc_G[k_idx] * invCount;
                memHndl.sp_B[k_idx] = memHndl.acc_B[k_idx] * invCount;
            }
        }   
    }

    // --- 4. ENFORCE CONNECTIVITY (Phase 2 - Zero Allocation & Bitwise Packed) ---
    
    ArtMosaic::fillProcBuf(memHndl.CC, numPixels, -1);

    const A_long expectedSize = numPixels / k;
    const A_long MIN_SIZE = expectedSize >> 2; 

    constexpr A_long dx[4] = {-1,  1,  0,  0};
    constexpr A_long dy[4] = { 0,  0, -1,  1};

    A_long new_k = 0;

    for (A_long j = 0; j < height; j++)
    {
        for (A_long i = 0; i < width; i++)
        {
            const A_long startIdx = j * width + i;

            if (memHndl.CC[startIdx] == -1)
            {
                const A_long oldLabel = memHndl.L[startIdx];
                A_long adjLabel = -1;

                if (i > 0 && memHndl.CC[startIdx - 1] >= 0) {
                    adjLabel = memHndl.CC[startIdx - 1];
                } else if (j > 0 && memHndl.CC[startIdx - width] >= 0) {
                    adjLabel = memHndl.CC[startIdx - width];
                }

                A_long qStart = 0;
                A_long qEnd = 0;
                
                // --- BITWISE PACKING (Y in high 16 bits, X in low 16 bits) ---
                memHndl.bfs_Queue[qEnd++] = (j << 16) | i;
                memHndl.CC[startIdx] = new_k;

                while (qStart < qEnd)
                {
                    const A_long packed = memHndl.bfs_Queue[qStart++];
                    
                    // Instant Extraction (Replaces slow modulo/division)
                    const A_long cx = packed & 0xFFFF;
                    const A_long cy = packed >> 16;

                    for (A_long n = 0; n < 4; n++)
                    {
                        const A_long nx = cx + dx[n];
                        const A_long ny = cy + dy[n];

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            const A_long nIdx = ny * width + nx;
                            
                            if (memHndl.CC[nIdx] == -1 && memHndl.L[nIdx] == oldLabel)
                            {
                                memHndl.bfs_Queue[qEnd++] = (ny << 16) | nx;
                                memHndl.CC[nIdx] = new_k;
                            }
                        }
                    }
                } // End BFS

                // Size Check & Label Exhaustion Check
                if (qEnd <= MIN_SIZE || new_k >= k - 1)
                {
                    if (adjLabel >= 0)
                    {
                        for (A_long q = 0; q < qEnd; q++)
                        {
                            const A_long p = memHndl.bfs_Queue[q];
                            const A_long pIdx = (p >> 16) * width + (p & 0xFFFF);
                            memHndl.CC[pIdx] = adjLabel;
                        }
                    }
                    else
                    {
                        if (new_k < k) new_k++;
                    }
                }
                else
                {
                    new_k++;
                }
            }
        }
    }

    for (A_long i = 0; i < numPixels; i++) 
    {
        memHndl.L[i] = memHndl.CC[i];
    }
    k = new_k; 

    // --- 5. RE-AVERAGE FINAL COLORS (Phase 3) ---

    ArtMosaic::fillProcBuf(memHndl.acc_R, k, 0.0f);
    ArtMosaic::fillProcBuf(memHndl.acc_G, k, 0.0f);
    ArtMosaic::fillProcBuf(memHndl.acc_B, k, 0.0f);
    ArtMosaic::fillProcBuf(memHndl.acc_Count, k, 0);

    for (A_long i = 0; i < numPixels; i++) 
    {
        const A_long label = memHndl.L[i];
        if (label >= 0 && label < k) 
        {
            memHndl.acc_R[label] += memHndl.R_planar[i];
            memHndl.acc_G[label] += memHndl.G_planar[i];
            memHndl.acc_B[label] += memHndl.B_planar[i];
            memHndl.acc_Count[label]++;
        }
    }

    for (A_long k_idx = 0; k_idx < k; k_idx++) 
    {
        const int32_t count = memHndl.acc_Count[k_idx];
        if (count > 0) 
        {
            const float invCount = 1.0f / static_cast<float>(count);
            memHndl.sp_R[k_idx] = memHndl.acc_R[k_idx] * invCount;
            memHndl.sp_G[k_idx] = memHndl.acc_G[k_idx] * invCount;
            memHndl.sp_B[k_idx] = memHndl.acc_B[k_idx] * invCount;
        }
    }

    // --- 6. RENDER MOSAIC AND BORDERS ---
    
    for (A_long j = 0; j < height; j++)
    {
        const A_long rowOffset = j * width;
        for (A_long i = 0; i < width; i++)
        {
            const A_long idx = rowOffset + i;
            const A_long lVal = memHndl.L[idx];

            bool isBorder = false;

            if (i + 1 < width && memHndl.L[idx + 1] != lVal) 
            {
                isBorder = true;
            }
            else if (j + 1 < height && memHndl.L[idx + width] != lVal) 
            {
                isBorder = true;
            }

            if (isBorder)
            {
                memHndl.R_planar[idx] = GrayColor.r; 
                memHndl.G_planar[idx] = GrayColor.g;
                memHndl.B_planar[idx] = GrayColor.b;
            }
            else if (lVal >= 0 && lVal < k)
            {
                memHndl.R_planar[idx] = memHndl.sp_R[lVal];
                memHndl.G_planar[idx] = memHndl.sp_G[lVal];
                memHndl.B_planar[idx] = memHndl.sp_B[lVal];
            }
            else
            {
                memHndl.R_planar[idx] = WhiteColor.r;
                memHndl.G_planar[idx] = WhiteColor.g;
                memHndl.B_planar[idx] = WhiteColor.b;
            }
        }
    }
    
    return;
}