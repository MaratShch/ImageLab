#include <cstdint>
#include "Common.hpp"
#include "AlgoControls.hpp"

// Calculate maximum possible elements in our window (Radius 8 = 17x17 = 289 elements)
constexpr int32_t MAX_WINDOW_AREA = (kernelRadiusMax * 2 + 1) * 
                                    (kernelRadiusMax * 2 + 1);


inline void AnalyzeAdaptiveWindow
(
    const float* RESTRICT centerPtr, 
    const int32_t strideElements, 
    const int32_t maxRadius,
    int32_t& outBestRadius, 
    float& outMin, 
    float& outMax
)
{
    // Fast local stack allocation (instantaneous, no heap fragmentation)
    CACHE_ALIGN float window[MAX_WINDOW_AREA];

    for (int32_t r = 1; r <= maxRadius; ++r) 
    {
        int32_t count = 0;
        outMin = 255.0f; // Max possible Y value in 8-bit equivalent
        outMax = 0.0f;   // Min possible Y value

        // 1. Extract the current window
        for (int32_t wy = -r; wy <= r; ++wy) 
        {
            const float* RESTRICT rowPtr = centerPtr + (wy * strideElements);
            for (int32_t wx = -r; wx <= r; ++wx) 
            {
                const float val = rowPtr[wx];
                window[count++] = val;
                
                // Track absolute min and max
                if (val < outMin) outMin = val;
                if (val > outMax) outMax = val;
            }
        }

        // 2. Find the mathematical median
        // PERFORMANCE TRICK: std::nth_element is O(N). It perfectly isolates 
        // the median without wasting CPU cycles doing a full O(N log N) sort!
        const int32_t halfIdx = count / 2;
        std::nth_element(window, window + halfIdx, window + count);
        const float currentMedian = window[halfIdx];

        // 3. Evaluate Median Validity
        // If the median itself is NOT an impulse (meaning it is greater than the 
        // local min and less than the local max), this window is structurally valid.
        if (currentMedian > outMin && currentMedian < outMax) 
        {
            outBestRadius = r;
            return; // Exit early, we found the perfect window size!
        }
    }
    
    // If we reach here, we hit maxRadius. We must settle for this window.
    outBestRadius = maxRadius;
}

template<typename CorruptedFunc>
inline float CalculateFrequencyMedian
(
    const float* RESTRICT centerPtr, 
    const int32_t strideElements, 
    const int32_t radius, 
    const float localMin, 
    const float localMax, 
    const float fallbackPixel,
    CorruptedFunc&& isCorrupted // Perfectly forwarded C++14 lambda
)
{
    // ========================================================================
    // 1. Extract Valid Pixels
    // ========================================================================
    float validPixels[MAX_WINDOW_AREA];
    int32_t validCount = 0;

    for (int32_t wy = -radius; wy <= radius; ++wy) 
    {
        const float* RESTRICT rowPtr = centerPtr + (wy * strideElements);
        for (int32_t wx = -radius; wx <= radius; ++wx) 
        {
            const float val = rowPtr[wx];
            
            // Execute the inlined lambda. If it is NOT noise, keep it.
            if (!isCorrupted(val, localMin, localMax)) 
            {
                validPixels[validCount++] = val;
            }
        }
    }

    // Fast exits if the window is completely destroyed or has only 1 valid pixel
    if (validCount == 0) return fallbackPixel;
    if (validCount == 1) return validPixels[0];

    // ========================================================================
    // 2. Sort the Valid Pixels
    // ========================================================================
    // We must fully sort here because deduplication requires 
    // identical values to sit sequentially in memory.
    std::sort(validPixels, validPixels + validCount);

    // ========================================================================
    // 3. Deduplicate and Count Frequencies
    // ========================================================================
    float uniqueValues[MAX_WINDOW_AREA];
    int32_t frequencies[MAX_WINDOW_AREA];
    int32_t uniqueCount = 0;

    // Initialize the first element
    uniqueValues[0] = validPixels[0];
    frequencies[0]  = 1;
    uniqueCount     = 1;

    for (int32_t i = 1; i < validCount; ++i) 
    {
        // Because floats were directly copied from the same memory source, 
        // strict equality (==) is mathematically safe and exact.
        if (validPixels[i] == uniqueValues[uniqueCount - 1]) 
        {
            frequencies[uniqueCount - 1]++;
        } 
        else 
        {
            uniqueValues[uniqueCount] = validPixels[i];
            frequencies[uniqueCount]  = 1;
            uniqueCount++;
        }
    }

    // ========================================================================
    // 4. Calculate the Final Structural Median
    // ========================================================================
    if (uniqueCount % 2 != 0) 
    {
        // ODD length: There is a mathematically perfect middle element.
        return uniqueValues[uniqueCount / 2];
    } 

    // EVEN length: We have two middle candidates.
    const int32_t midIdx1 = (uniqueCount / 2) - 1;
    const int32_t midIdx2 = uniqueCount / 2;

    const int32_t freq1 = frequencies[midIdx1];
    const int32_t freq2 = frequencies[midIdx2];

    // TIE-BREAKER: Return the pixel value that occurs more frequently in the 
    // surrounding image structure.
    if (freq1 > freq2) 
    {
        return uniqueValues[midIdx1];
    } 
    
    if (freq2 > freq1) 
    {
        return uniqueValues[midIdx2];
    } 

    // Absolute Tie: Average the two structural candidates to prevent bias.
    // This acts as the absolute final catch-all return path for the compiler.
    return (uniqueValues[midIdx1] + uniqueValues[midIdx2]) * 0.5f;
}



void ProcessImage_Scalar
(
    const float* RESTRICT inY, 
    float* RESTRICT outY, 
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t strideY_Elements, 
    const AlgoControls& ctrl
)
{
    // ========================================================================
    // LAMBDA: Noise Classification (Block B)
    // C++14 Init-Capture: We extract tolerance once and bake it into the lambda.
    // ========================================================================
    auto isPixelCorrupted = [tol = ctrl.tolerance](const float center, const float lMin, const float lMax) -> bool 
    {
        return (center <= (lMin + tol)) || (center >= (lMax - tol));
    };

    // ==========================================
    // Outer Spatial Loop (Rows)
    // ==========================================
    for (int32_t y = 0; y < sizeY; ++y) 
    {
        const float* RESTRICT inRow  = inY + (y * strideY_Elements);
        float* RESTRICT outRow = outY + (y * strideY_Elements);

        // ==========================================
        // Inner Spatial Loop (Columns)
        // ==========================================
        for (int32_t x = 0; x < sizeX; ++x) 
        {
            const float centerPixel = inRow[x];
            const float* centerPtr  = &inRow[x];

            // ----------------------------------------------------
            // BLOCK A: Analyze Adaptive Window
            // ----------------------------------------------------
            int32_t bestRadius = 1;
            float localMin = 0.0f;
            float localMax = 0.0f;

            AnalyzeAdaptiveWindow
            (
                centerPtr, strideY_Elements, ctrl.radius, 
                bestRadius, localMin, localMax
            );

            // ----------------------------------------------------
            // BLOCK B: Execute the Lambda
            // ----------------------------------------------------
            // Notice how clean the signature is now. We only pass the dynamically 
            // changing variables. The tolerance is baked into the lambda's state.
            if (!isPixelCorrupted(centerPixel, localMin, localMax)) 
            {
                outRow[x] = centerPixel;
                continue; // Fast-exit
            }

			// ----------------------------------------------------
            // BLOCK C: Frequency Median Solver
            // ----------------------------------------------------
            // We pass the lambda `isPixelCorrupted` directly into the function
            float replacementPixel = 
			CalculateFrequencyMedian
			(
                centerPtr, 
                strideY_Elements, 
                bestRadius, 
                localMin, 
                localMax, 
                centerPixel,      // Fallback
                isPixelCorrupted  // The C++14 Lambda
            );

            outRow[x] = replacementPixel;
        }
    }
}