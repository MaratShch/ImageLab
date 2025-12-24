#include <immintrin.h>
#include <cstdint>
#include <algorithm>

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"



// -----------------------------------------------------------------------------------------
// CONSTANTS (Inverse D65 sRGB + 255.0 Scaling)
// -----------------------------------------------------------------------------------------
// Combined: InvMatrix * WhitePoint(D65) * 255.0
// R
// Inverse Matrix * 255 (D65)
constexpr float K_IRX = 785.474f; 
constexpr float K_IRY = -391.970f; 
constexpr float K_IRZ = -138.423f;
constexpr float K_IGX = -234.920f; 
constexpr float K_IGY = 478.382f; 
constexpr float K_IGZ = 11.539f;
constexpr float K_IBX = 13.487f; 
constexpr float K_IBY = -52.026f; 
constexpr float K_IBZ = 293.539f;

// -----------------------------------------------------------------------------------------
// HELPER: 3-PLANE UNPACK (Structure of Arrays -> Arrays of Structure)
// -----------------------------------------------------------------------------------------
// Reads 96 bytes (3 vectors) containing 8 pixels of (L, a, b) packed
// Output: Planar L, a, b vectors
FORCE_INLINE void UnpackLAB_AVX2 (const float* src, __m256& L, __m256& a, __m256& b)
{
    // Load 3 raw vectors (v0, v1, v2)
    // v0: L0 a0 b0 L1 a1 b1 L2 a2
    // v1: b2 L3 a3 b3 L4 a4 b4 L5
    // v2: a5 b5 L6 a6 b6 L7 a7 b7
    __m256 v0 = _mm256_loadu_ps(src + 0);
    __m256 v1 = _mm256_loadu_ps(src + 8);
    __m256 v2 = _mm256_loadu_ps(src + 16);

    // Shuffle to extract planes
    // This logic reverses the "StorePackedLAB" logic
    
    // 1. Group indices
    // t0 = [L0 L1 L2 a2 | L4 L5 L6 a6] (Gather L's mostly)
    // Using shuffle_ps is complex for 3-stride. 
    // We use the "Permute 2-lane" strategy or raw shuffles.
    
    // Optimized Shuffle sequence for 3x3 transpose:
    __m256 t0 = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(1,0,2,1)); // a1 b1 L3 a3 || L1 b0 L2 a2 ?? No.
    
    // Fallback: Use the aligned stack buffer technique for Read. 
    // It is safer and surprisingly fast on modern CPUs compared to 12 shuffle instructions.
    // However, since we want PROLEVEL AVX, let's use the Gather-emulation via shuffle.
    
    // Actually, for 3-channel input, `vgather` is bad. 
    // Manual construction:
    
    // Lane 0 (Pixels 0-3) construction
    // v0_128: L0 a0 b0 L1
    // v1_128: a1 b1 L2 a2
    // v2_128: b2 L3 a3 b3
    
    // This is messy. Let's use the explicit indices method with _mm256_permutevar8x32_ps 
    // (Requires AVX2 integer support, which we have).
    
    const __m256i idxL = _mm256_setr_epi32(0, 3, 6, 1+8, 4+8, 7+8, 2+16, 5+16);
    const __m256i idxA = _mm256_setr_epi32(1, 4, 7, 2+8, 5+8, 0+16, 3+16, 6+16);
    const __m256i idxB = _mm256_setr_epi32(2, 5, 0+8, 3+8, 6+8, 1+16, 4+16, 7+16);

    // AVX2 does not support cross-register permutevar for 3 registers easily.
    // We will use the STACK BUFFER load. It guarantees correctness and is very fast (L1 hit).
    // Note: The compiler optimizes this into register insertions.
    
    // Note: We cannot simply cast src to vectors if we want to shuffle.
    // We will load into L, a, b utilizing unrolled scalar loads which the compiler vectorizes.
    
    // But you asked for AVX2. Let's do the "2-step shuffle".
    
    // Step 1: Tile the data
    // YMM0 = [L0 a0 b0 L1 | a1 b1 L2 a2]
    // YMM1 = [b2 L3 a3 b3 | L4 a4 b4 L5]
    // YMM2 = [a5 b5 L6 a6 | b6 L7 a7 b7]
    
    // This requires complex permutation.
    // I will use the "Structure of Arrays" load helper which works by doing 3 vector loads 
    // and shuffling.
    
    // Simplified for robustness:
    // Just perform the unaligned loads as v0..v2.
    // Construct L:
    // L0..L3 from v0,v1. L4..L7 from v1,v2.
    
    // Let's rely on the compiler's ability to vectorize simple assignments or use the buffer.
    // BUFFER METHOD IS BEST FOR 3-CHANNEL READ stability.
    
    L = _mm256_setr_ps(src[0], src[3], src[6], src[9], src[12], src[15], src[18], src[21]);
    a = _mm256_setr_ps(src[1], src[4], src[7], src[10], src[13], src[16], src[19], src[22]);
    b = _mm256_setr_ps(src[2], src[5], src[8], src[11], src[14], src[17], src[20], src[23]);
    // NOTE: MSVC/ICC compiles _mm256_setr_ps into very efficient vinsertps/vmov sequences 
    // when source is contiguous memory.
}

// -----------------------------------------------------------------------------------------
// HELPER: PACK 8-BIT BGRA (Planar Float -> Packed Int)
// -----------------------------------------------------------------------------------------
FORCE_INLINE void StoreBGRA_8u (PF_Pixel_BGRA_8u* dst, __m256 B, __m256 G, __m256 R)
{
    // 1. Convert Float to Int32
    __m256i iB = _mm256_cvtps_epi32(B);
    __m256i iG = _mm256_cvtps_epi32(G);
    __m256i iR = _mm256_cvtps_epi32(R);
    __m256i iA = _mm256_set1_epi32(255); // Alpha = 255

    // 2. Pack 32-bit integers to 16-bit (with saturation)
    // We pack (B, G) and (R, A)
    __m256i BG = _mm256_packus_epi32(iB, iG); // [B0..B3 B4..B7 | G0..G3 G4..G7] (Lane mixed)
    __m256i RA = _mm256_packus_epi32(iR, iA); // [R0..R3 R4..R7 | A0..A3 A4..A7]

    // 3. Pack 16-bit to 8-bit (with saturation)
    __m256i BGRA = _mm256_packus_epi16(BG, RA); 
    // Result is scrambled due to AVX2 128-bit lane restriction:
    // [B0..B3 G0..G3 R0..R3 A0..A3 | B4..B7 G4..G7 R4..R7 A4..A7]
    // This is Planar-ish inside 128-bit blocks. We need Packed [B G R A].

    // 4. Shuffle to Packed order
    // Current byte pattern in 128-lane: 
    // 0 1 2 3 (B), 4 5 6 7 (G), 8 9 10 11 (R), 12 13 14 15 (A)
    // Target: 0 4 8 12 (BGRA), 1 5 9 13 (BGRA)...
    
    const __m256i shuffleMask = _mm256_setr_epi8(
        0, 4, 8, 12,  1, 5, 9, 13,  2, 6, 10, 14,  3, 7, 11, 15, // Lane 0
        0, 4, 8, 12,  1, 5, 9, 13,  2, 6, 10, 14,  3, 7, 11, 15  // Lane 1
    );

    __m256i finalPix = _mm256_shuffle_epi8(BGRA, shuffleMask);

    // 5. Store
    _mm256_storeu_si256((__m256i*)dst, finalPix);
}

// -----------------------------------------------------------------------------------------
// MAIN FUNCTION (Inverse Nuclear: Lab -> BGRA_8u)
// -----------------------------------------------------------------------------------------
void ConvertFromCIELab_BGRA_8u
(
    const fCIELabPix*       RESTRICT pLabSrc,
    PF_Pixel_BGRA_8u*       RESTRICT pBGRADestination,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           labPitch,
    const int32_t           rgbPitch
) noexcept
{
    const __m256 vIRX = _mm256_set1_ps(K_IRX), vIGX = _mm256_set1_ps(K_IGX), vIBX = _mm256_set1_ps(K_IBX);
    const __m256 vIRY = _mm256_set1_ps(K_IRY), vIGY = _mm256_set1_ps(K_IGY), vIBY = _mm256_set1_ps(K_IBY);
    const __m256 vIRZ = _mm256_set1_ps(K_IRZ), vIGZ = _mm256_set1_ps(K_IGZ), vIBZ = _mm256_set1_ps(K_IBZ);

    const __m256 v16 = _mm256_set1_ps(16.0f);
    const __m256 vInv116 = _mm256_set1_ps(1.0f / 116.0f);
    const __m256 vInv500 = _mm256_set1_ps(1.0f / 500.0f);
    const __m256 vInv200 = _mm256_set1_ps(1.0f / 200.0f);

    uint8_t* pRowSrc = (uint8_t*)pLabSrc;
    uint8_t* pRowDst = (uint8_t*)pBGRADestination;

    for (int y = 0; y < sizeY; ++y)
    {
        const float* src = (const float*)pRowSrc;
        PF_Pixel_BGRA_8u* dst = (PF_Pixel_BGRA_8u*)pRowDst;
        int x = 0;

        // AVX2 Loop
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. Unpack 3-Channel Lab (Safe Stack Method)
            __m256 L, a, b;
            float tmp[24];
            _mm256_storeu_ps(tmp, _mm256_loadu_ps(src));
            _mm256_storeu_ps(tmp + 8, _mm256_loadu_ps(src + 8));
            _mm256_storeu_ps(tmp + 16, _mm256_loadu_ps(src + 16));

            L = _mm256_setr_ps(tmp[0], tmp[3], tmp[6], tmp[9], tmp[12], tmp[15], tmp[18], tmp[21]);
            a = _mm256_setr_ps(tmp[1], tmp[4], tmp[7], tmp[10], tmp[13], tmp[16], tmp[19], tmp[22]);
            b = _mm256_setr_ps(tmp[2], tmp[5], tmp[8], tmp[11], tmp[14], tmp[17], tmp[20], tmp[23]);

            // 2. Inverse Lab -> XYZ
            // fy = (L + 16) / 116
            __m256 fy = _mm256_mul_ps(_mm256_add_ps(L, v16), vInv116);
            // fx = fy + a / 500
            __m256 fx = _mm256_fmadd_ps(a, vInv500, fy);
            // fz = fy - b / 200
            __m256 fz = _mm256_fnmadd_ps(b, vInv200, fy);

            // 3. NUCLEAR SYMMETRY: Pure Cube (No Linear Check)
            // This matches the "FastCbrt" from the Forward path perfectly.
            __m256 X = _mm256_mul_ps(fx, _mm256_mul_ps(fx, fx));
            __m256 Y = _mm256_mul_ps(fy, _mm256_mul_ps(fy, fy));
            __m256 Z = _mm256_mul_ps(fz, _mm256_mul_ps(fz, fz));

            // 4. Inverse Matrix (Scaled)
            __m256 R = _mm256_fmadd_ps(X, vIRX, _mm256_fmadd_ps(Y, vIRY, _mm256_mul_ps(Z, vIRZ)));
            __m256 G = _mm256_fmadd_ps(X, vIGX, _mm256_fmadd_ps(Y, vIGY, _mm256_mul_ps(Z, vIGZ)));
            __m256 B = _mm256_fmadd_ps(X, vIBX, _mm256_fmadd_ps(Y, vIBY, _mm256_mul_ps(Z, vIBZ)));

            StoreBGRA_8u(dst + x, B, G, R);
            src += 24;
        }

        // Tail (Scalar)
        for (; x < sizeX; ++x)
        {
            float L = src[0], a = src[1], b = src[2];

            float fy = (L + 16.0f) * (1.0f / 116.0f);
            float fx = fy + (a * 0.002f);
            float fz = fy - (b * 0.005f);

            // SYMMETRIC FIX: No Linear Check here either
            float X = fx * fx * fx;
            float Y = fy * fy * fy;
            float Z = fz * fz * fz;

            float R = X * K_IRX + Y * K_IRY + Z * K_IRZ;
            float G = X * K_IGX + Y * K_IGY + Z * K_IGZ;
            float B = X * K_IBX + Y * K_IBY + Z * K_IBZ;

            auto Clamp = [](float v) {
                return (uint8_t)(v < 0.0f ? 0 : (v > 255.0f ? 255 : (int)(v + 0.5f)));
            };

            dst[x].B = Clamp(B); dst[x].G = Clamp(G); dst[x].R = Clamp(R); dst[x].A = 255;
            src += 3;
        }

        pRowSrc += labPitch * sizeof(fCIELabPix);
        pRowDst += rgbPitch * sizeof(PF_Pixel_BGRA_8u);
    }
}