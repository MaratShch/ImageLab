#include <algorithm>
#include "ColorConvert.hpp"
#include "AlgoMemHandler.hpp"


void dispatch_convert_to_planar
(
    const PF_Pixel_BGRA_8u* imgInBuffer, 
    const MemHandler& memHndl, 
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t linePitch
)
{
    // ==========================================
    // REC. 709 Coefficients
    // ==========================================
    // Luma (Y)
    constexpr float R_Y =  0.212600f;
    constexpr float G_Y =  0.715200f;
    constexpr float B_Y =  0.072200f;

    // Chroma Blue (U / Cb) 
    constexpr float R_U = -0.114572f;
    constexpr float G_U = -0.385428f;
    constexpr float B_U =  0.500000f;

    // Chroma Red (V / Cr)
    constexpr float R_V =  0.500000f;
    constexpr float G_V = -0.454153f;
    constexpr float B_V = -0.045847f;

    // Retrieve the cache-aligned physical stride from our Arena
    const int32_t outStride = memHndl.strideY_Elements;

    // ==========================================
    // Conversion Loop (Scalar)
    // ==========================================
    for (int32_t y = 0; y < sizeY; ++y) 
    {
        // 1. Input pointer for the current row
        // linePitch is in pixels. Pointer arithmetic automatically scales by sizeof(PF_Pixel_BGRA_8u)
        const PF_Pixel_BGRA_8u* inRow = imgInBuffer + (y * linePitch);

        // 2. Output pointers for the current row
        float* outYRow = memHndl.proc_Y + (y * outStride);
        float* outURow = memHndl.proc_U + (y * outStride);
        float* outVRow = memHndl.proc_V + (y * outStride);

        for (int32_t x = 0; x < sizeX; ++x) 
        {
            // Extract channels directly using the struct members
            const float B = static_cast<float>(inRow[x].B);
            const float G = static_cast<float>(inRow[x].G);
            const float R = static_cast<float>(inRow[x].R);

            // Compute and write REC. 709 Planar YUV
            outYRow[x] = (R_Y * R) + (G_Y * G) + (B_Y * B);
            outURow[x] = (R_U * R) + (G_U * G) + (B_U * B);
            outVRow[x] = (R_V * R) + (G_V * G) + (B_V * B);
        }
    }
    
    return;
}


void dispatch_convert_to_interleaved
(
    const MemHandler& memHndl,
    const PF_Pixel_BGRA_8u* originalInBuffer, // Needed to copy the Alpha channel
    PF_Pixel_BGRA_8u* outBuffer,              // The final Adobe render destination
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t inLinePitchPixels,          // MUST be signed int32_t!
    const int32_t outLinePitchPixels          // MUST be signed int32_t!
)
{
    // ==========================================
    // Inverse REC. 709 Coefficients (YUV to RGB)
    // ==========================================
    constexpr float Y_R = 1.0f;
    constexpr float U_R = 0.0f;
    constexpr float V_R = 1.5748f;

    constexpr float Y_G = 1.0f;
    constexpr float U_G = -0.1873f;
    constexpr float V_G = -0.4681f;

    constexpr float Y_B = 1.0f;
    constexpr float U_B = 1.8556f;
    constexpr float V_B = 0.0f;

    const int32_t arenaStride = memHndl.strideY_Elements;

    // ==========================================
    // Conversion Loop (Scalar)
    // ==========================================
    for (int32_t y = 0; y < sizeY; ++y) 
    {
        // 1. Adobe Buffer Pointers (Safe for Negative Pitch)
        const PF_Pixel_BGRA_8u* inRowOriginal = originalInBuffer + (y * inLinePitchPixels);
        PF_Pixel_BGRA_8u* outRowAdobe         = outBuffer + (y * outLinePitchPixels);

        // 2. Arena Pointers 
        // We read from out_Y (the result of the 1st iteration) and the untouched U/V
        const float* inYRow = memHndl.out_Y  + (y * arenaStride);
        const float* inURow = memHndl.proc_U + (y * arenaStride);
        const float* inVRow = memHndl.proc_V + (y * arenaStride);

        for (int32_t x = 0; x < sizeX; ++x) 
        {
            // Extract the planar float values
            const float Y = inYRow[x];
            const float U = inURow[x];
            const float V = inVRow[x];

            // Calculate inverse REC. 709
            float R = (Y_R * Y) + (U_R * U) + (V_R * V);
            float G = (Y_G * Y) + (U_G * U) + (V_G * V);
            float B = (Y_B * Y) + (U_B * U) + (V_B * V);

            // C++14 Clamping (0.0f to 255.0f)
            R = std::max(0.0f, std::min(255.0f, R));
            G = std::max(0.0f, std::min(255.0f, G));
            B = std::max(0.0f, std::min(255.0f, B));

            // Write back to the Adobe struct
            outRowAdobe[x].R = static_cast<A_u_char>(R + 0.5f); // +0.5f for rounding
            outRowAdobe[x].G = static_cast<A_u_char>(G + 0.5f);
            outRowAdobe[x].B = static_cast<A_u_char>(B + 0.5f);
            
            // Preserve the original Alpha channel
            outRowAdobe[x].A = inRowOriginal[x].A; 
        }
    }
}