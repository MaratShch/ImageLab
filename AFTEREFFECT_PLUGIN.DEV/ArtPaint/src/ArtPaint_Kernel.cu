#include "ArtPaint_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <algorithm>
#include <cmath>

// ============================================================================
// INTERNAL ARENA MANAGER
// ============================================================================
static void* g_arenaMem = nullptr;
static size_t g_currentArenaSize = 0;

static int g_cachedWidth = 0;
static int g_cachedHeight = 0;
static RenderQuality g_cachedQuality = RenderQuality::Fast_HalfSize;
static int g_cachedRadius = 0;

constexpr size_t CUDA_ALIGNMENT = 256ull;
inline constexpr size_t AlignSizeCuda(const size_t size) noexcept
{
    return (size + CUDA_ALIGNMENT - 1) & ~(CUDA_ALIGNMENT - 1);
}

static size_t GetRequiredBytes(int width, int height, RenderQuality quality, int radius) noexcept
{
    int proc_width = (quality == RenderQuality::Fast_HalfSize) ? (width >> 1) : width;
    int proc_height = (quality == RenderQuality::Fast_HalfSize) ? (height >> 1) : height;

    size_t frameSize = static_cast<size_t>(proc_width) * static_cast<size_t>(proc_height);
    size_t alignedFloatSize = AlignSizeCuda(frameSize * sizeof(float));

    const size_t max_local_edges = (2 * radius + 1) * 2;
    const size_t max_edges = frameSize * max_local_edges;

    size_t alignedEdgeIdxSize = AlignSizeCuda(max_edges * sizeof(int));

    size_t totalBytes = 0;
    // Buffers: Y, U, V, tA, tB, tC, tmpA, tmpB, tmpC, tAsm, tBsm, tCsm, imProc1, imProc2 = 14 buffers
    totalBytes += 14 * alignedFloatSize;
    totalBytes += 2 * alignedEdgeIdxSize; // I, J arrays

    return totalBytes;
}

static bool EnsureArenaMemory(int width, int height, RenderQuality quality, int radius) noexcept
{
    if (g_arenaMem != nullptr && g_cachedWidth == width && g_cachedHeight == height &&
        g_cachedQuality == quality && g_cachedRadius == radius) return true;

    size_t requiredBytes = GetRequiredBytes(width, height, quality, radius);

    if (requiredBytes > g_currentArenaSize)
    {
        cudaDeviceSynchronize();
        if (g_arenaMem != nullptr) { cudaFree(g_arenaMem); g_arenaMem = nullptr; }

        cudaError_t err = cudaMalloc(&g_arenaMem, requiredBytes);
        if (err != cudaSuccess) return false;

        g_currentArenaSize = requiredBytes;
    }

    g_cachedWidth = width; g_cachedHeight = height;
    g_cachedQuality = quality; g_cachedRadius = radius;
    return true;
}

void FreeArtPaintArena() noexcept
{
    if (g_arenaMem != nullptr) { cudaDeviceSynchronize(); cudaFree(g_arenaMem); g_arenaMem = nullptr; g_currentArenaSize = 0; }
}

inline __device__ float4 HalfToFloat4(const Pixel16& in) noexcept
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(const float4& in) noexcept
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}

// ============================================================================
// DEVICE CONSTANTS
// ============================================================================
__constant__ float c_y_r = 0.2126f, c_y_g = 0.7152f, c_y_b = 0.0722f;
__constant__ float c_u_r = -0.114572f, c_u_g = -0.385428f, c_u_b = 0.5f;
__constant__ float c_v_r = 0.5f, c_v_g = -0.454153f, c_v_b = -0.045847f;
__constant__ float c_inv_r_v = 1.5748f, c_inv_g_u = -0.187324f, c_inv_g_v = -0.468124f, c_inv_b_u = 1.8556f;

// ============================================================================
// TEMPLATED ADOBE I/O (Phase 0)
// ============================================================================
__device__ __forceinline__ float4 ReadAdobePixel(const void* __restrict__ buffer, int pitch, int x, int y, bool is16f)
{
    if (is16f) return HalfToFloat4(((const Pixel16*)buffer)[y * pitch + x]);
    return ((const float4*)buffer)[y * pitch + x];
}

__device__ __forceinline__ void WriteAdobePixel(void* __restrict__ buffer, int pitch, int x, int y, float4 color, bool is16f)
{
    if (is16f) ((Pixel16*)buffer)[y * pitch + x] = FloatToHalf4(color);
    else ((float4*)buffer)[y * pitch + x] = color;
}

template <bool IS_HALF>
__global__ void Kernel_AdobeToPlanar(const void* __restrict__ inBuffer, int srcPitch, float* __restrict__ outY, float* __restrict__ outU, float* __restrict__ outV, int outWidth, int outHeight, int inWidth, int inHeight, bool is16f)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= outWidth || out_y >= outHeight) return;
    const int idx = out_y * outWidth + out_x;

    if (IS_HALF)
    {
        const int in_x = out_x * 2, in_y = out_y * 2;
        const int in_x1 = min(in_x + 1, inWidth - 1), in_y1 = min(in_y + 1, inHeight - 1);
        auto extract_color = [](float4 p, float& r, float& g, float& b) { b = p.x * 255.0f; g = p.y * 255.0f; r = p.z * 255.0f; };
        float r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
        extract_color(ReadAdobePixel(inBuffer, srcPitch, in_x, in_y, is16f), r00, g00, b00);
        extract_color(ReadAdobePixel(inBuffer, srcPitch, in_x1, in_y, is16f), r01, g01, b01);
        extract_color(ReadAdobePixel(inBuffer, srcPitch, in_x, in_y1, is16f), r10, g10, b10);
        extract_color(ReadAdobePixel(inBuffer, srcPitch, in_x1, in_y1, is16f), r11, g11, b11);

        outY[idx] = c_y_r * ((r00 + r01 + r10 + r11) * 0.25f) + c_y_g * ((g00 + g01 + g10 + g11) * 0.25f) + c_y_b * ((b00 + b01 + b10 + b11) * 0.25f);
        outU[idx] = c_u_r * ((r00 + r01 + r10 + r11) * 0.25f) + c_u_g * ((g00 + g01 + g10 + g11) * 0.25f) + c_u_b * ((b00 + b01 + b10 + b11) * 0.25f);
        outV[idx] = c_v_r * ((r00 + r01 + r10 + r11) * 0.25f) + c_v_g * ((g00 + g01 + g10 + g11) * 0.25f) + c_v_b * ((b00 + b01 + b10 + b11) * 0.25f);
    }
    else
    {
        float4 px = ReadAdobePixel(inBuffer, srcPitch, out_x, out_y, is16f);
        const float b = px.x * 255.0f, g = px.y * 255.0f, r = px.z * 255.0f;
        outY[idx] = (c_y_r * r) + (c_y_g * g) + (c_y_b * b);
        outU[idx] = (c_u_r * r) + (c_u_g * g) + (c_u_b * b);
        outV[idx] = (c_v_r * r) + (c_v_g * g) + (c_v_b * b);
    }
}

template <bool IS_HALF>
__global__ void Kernel_PlanarToAdobe(const void* __restrict__ origBuffer, int origPitch, void* __restrict__ outBuffer, int dstPitch, const float* __restrict__ inY, const float* __restrict__ inU, const float* __restrict__ inV, int outWidth, int outHeight, int planarWidth, int planarHeight, bool is16f)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= outWidth || out_y >= outHeight) return;

    float r, g, b;
    if (IS_HALF)
    {
        const float src_x = (static_cast<float>(out_x) + 0.5f) * 0.5f - 0.5f;
        const float src_y = (static_cast<float>(out_y) + 0.5f) * 0.5f - 0.5f;
        int px = static_cast<int>(floorf(src_x)), py = static_cast<int>(floorf(src_y));
        const float fx = src_x - px, fy = src_y - py;
        const float w00 = (1.0f - fx) * (1.0f - fy), w01 = fx * (1.0f - fy), w10 = (1.0f - fx) * fy, w11 = fx * fy;
        const int px0 = max(0, min(px, planarWidth - 1)), px1 = max(0, min(px + 1, planarWidth - 1));
        const int py0 = max(0, min(py, planarHeight - 1)), py1 = max(0, min(py + 1, planarHeight - 1));

        auto fetch_bilinear = [&](const float* ch) { return ch[py0 * planarWidth + px0] * w00 + ch[py0 * planarWidth + px1] * w01 + ch[py1 * planarWidth + px0] * w10 + ch[py1 * planarWidth + px1] * w11; };

        r = fmaxf(0.0f, fminf(255.0f, fetch_bilinear(inY) + (c_inv_r_v * fetch_bilinear(inV)))) * 0.0039215686f;
        g = fmaxf(0.0f, fminf(255.0f, fetch_bilinear(inY) + (c_inv_g_u * fetch_bilinear(inU)) + (c_inv_g_v * fetch_bilinear(inV)))) * 0.0039215686f;
        b = fmaxf(0.0f, fminf(255.0f, fetch_bilinear(inY) + (c_inv_b_u * fetch_bilinear(inU)))) * 0.0039215686f;
    }
    else
    {
        const int idx = out_y * outWidth + out_x;
        r = fmaxf(0.0f, fminf(255.0f, inY[idx] + (c_inv_r_v * inV[idx]))) * 0.0039215686f;
        g = fmaxf(0.0f, fminf(255.0f, inY[idx] + (c_inv_g_u * inU[idx]) + (c_inv_g_v * inV[idx]))) * 0.0039215686f;
        b = fmaxf(0.0f, fminf(255.0f, inY[idx] + (c_inv_b_u * inU[idx]))) * 0.0039215686f;
    }
    float4 origPx = ReadAdobePixel(origBuffer, origPitch, out_x, out_y, is16f);
    WriteAdobePixel(outBuffer, dstPitch, out_x, out_y, make_float4(b, g, r, origPx.w), is16f);
}

// ============================================================================
// VECTORIZED PHASE 1 & 2 (Tensors & Smoothing)
// ============================================================================
__global__ void Kernel_ComputeStructureTensors(const float* __restrict__ inY, float* __restrict__ tA, float* __restrict__ tB, float* __restrict__ tC, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float left = inY[y * width + max(x - 1, 0)], right = inY[y * width + min(x + 1, width - 1)];
    const float up = inY[max(y - 1, 0) * width + x], down = inY[min(y + 1, height - 1) * width + x];
    const float gX = (right - left) * 0.5f, gY = (down - up) * 0.5f;
    const int idx = y * width + x;
    tA[idx] = gX * gX; tB[idx] = gY * gY; tC[idx] = gX * gY;
}

__global__ void Kernel_Blur_Horizontal_Vec(const float* __restrict__ inA, const float* __restrict__ inB, const float* __restrict__ inC, float* __restrict__ outA, float* __restrict__ outB, float* __restrict__ outC, int width, int height, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sumA = 0.0f, sumB = 0.0f, sumC = 0.0f; int count = 0;
    for (int k = -radius; k <= radius; ++k) {
        const int px = max(0, min(x + k, width - 1));
        const int idx = y * width + px;
        sumA += inA[idx]; sumB += inB[idx]; sumC += inC[idx]; count++;
    }
    const int out_idx = y * width + x;
    const float inv_c = 1.0f / static_cast<float>(count);
    outA[out_idx] = sumA * inv_c; outB[out_idx] = sumB * inv_c; outC[out_idx] = sumC * inv_c;
}

__global__ void Kernel_Blur_Vertical_Vec(const float* __restrict__ inA, const float* __restrict__ inB, const float* __restrict__ inC, float* __restrict__ outA, float* __restrict__ outB, float* __restrict__ outC, int width, int height, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sumA = 0.0f, sumB = 0.0f, sumC = 0.0f; int count = 0;
    for (int k = -radius; k <= radius; ++k) {
        const int py = max(0, min(y + k, height - 1));
        const int idx = py * width + x;
        sumA += inA[idx]; sumB += inB[idx]; sumC += inC[idx]; count++;
    }
    const int out_idx = y * width + x;
    const float inv_c = 1.0f / static_cast<float>(count);
    outA[out_idx] = sumA * inv_c; outB[out_idx] = sumB * inv_c; outC[out_idx] = sumC * inv_c;
}

// ============================================================================
// FUSED PHASE 3 & 4 (Eigenvectors + Graph Builder)
// ============================================================================
__global__ void Kernel_BuildEdgeGraph_Fused
(
    const float* __restrict__ tensorA_sm, const float* __restrict__ tensorB_sm, const float* __restrict__ tensorC_sm,
    int* __restrict__ I, int* __restrict__ J,
    int width, int height, int radius,
    int max_local_edges, float angle_tol_sq, float cos_rot, float sin_rot
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int i = y * width + x;
    const int chunk_start = i * max_local_edges;

    // 1. Fill Sentinels
    for (int k = 0; k < max_local_edges; ++k) I[chunk_start + k] = -1;

    // 2. Compute Local Eigenvector in Registers
    const float A = tensorA_sm[i], B = tensorB_sm[i], C = tensorC_sm[i];
    const float trace = A + B;
    const float disc = sqrtf((A - B) * (A - B) + 4.0f * C * C);
    const float l1 = (trace + disc) * 0.5f;

    float vx = l1 - B, vy = C;
    if (fabsf(C) < 1e-6f) { vx = (A > B) ? 1.0f : 0.0f; vy = (A > B) ? 0.0f : 1.0f; }
    const float mag = sqrtf(vx * vx + vy * vy);
    float ex_orig = (mag > 1e-6f) ? (vx / mag) : 0.0f;
    float ey_orig = (mag > 1e-6f) ? (vy / mag) : 0.0f;

    // 3. Apply Angle Transformation
    const float ex = ex_orig * cos_rot - ey_orig * sin_rot;
    const float ey = ex_orig * sin_rot + ey_orig * cos_rot;

    // 4. Neighborhood Search
    const int radius_sq = radius * radius;
    int local_edges = 0;
    bool is_full = false;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            if (local_edges >= max_local_edges) { is_full = true; break; }
            if (dx == 0 && dy == 0) continue;

            const int len_sq = dx * dx + dy * dy;
            if (len_sq > radius_sq) continue;

            const int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                const float dot = ex * static_cast<float>(dx) + ey * static_cast<float>(dy);
                if (dot > 0.0f && (dot * dot) >= angle_tol_sq * static_cast<float>(len_sq))
                {
                    I[chunk_start + local_edges] = i;
                    J[chunk_start + local_edges] = ny * width + nx;
                    local_edges++;
                }
            }
        }
        if (is_full) break;
    }
}

// ============================================================================
// TEMPLATED MORPHOLOGY KERNEL (Phase 5)
// ============================================================================
__device__ __forceinline__ void atomicMinFloat(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMaxFloat(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

template <bool IS_DILATE>
__global__ void Kernel_Morph(const float* __restrict__ imIn, float* __restrict__ imOut, const int* __restrict__ I, const int* __restrict__ J, int frameSize, int max_local_edges, float bias)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frameSize) return;

    const int chunk_start = i * max_local_edges;
    if (I[chunk_start] < 0) return;

    const float imIn_i = imIn[i];
    float local_val = imIn_i;

    for (int k = 0; k < max_local_edges; ++k)
    {
        if (I[chunk_start + k] < 0) break;

        const int j = J[chunk_start + k];
        const float imIn_j = imIn[j];

        if (IS_DILATE)
        {
            atomicMaxFloat(&imOut[j], imIn_i + bias);
            local_val = fmaxf(local_val, imIn_j + bias);
        }
        else
        {
            atomicMinFloat(&imOut[j], imIn_i - bias);
            local_val = fminf(local_val, imIn_j - bias);
        }
    }

    if (IS_DILATE) atomicMaxFloat(&imOut[i], local_val);
    else atomicMinFloat(&imOut[i], local_val);
}

// ============================================================================
// HOST DISPATCHER
// ============================================================================
CUDA_KERNEL_CALL
void ArtPaint_CUDA
(
    const void* RESTRICT inBuffer, void* RESTRICT outBuffer,
    int srcPitch, int dstPitch, int width, int height,
    const AlgoControls* algoGpuParams, int frameCounter, bool isFloat16, cudaStream_t stream
)
{
    const StrokeBias    bias = algoGpuParams->bias;
    const RenderQuality quality = algoGpuParams->quality;
    float               sigma = algoGpuParams->sigma;
    float               bias_val = static_cast<float>(algoGpuParams->bias);
    const float         angular = algoGpuParams->angular;
    const float         angle = algoGpuParams->angle;
    const int32_t       iter = algoGpuParams->iter;

    if (quality == RenderQuality::Fast_HalfSize) sigma *= 0.5f;
    if (sigma > 25.0f) sigma = 25.0f; // Global safety cap
    const int radius = max(1, static_cast<int>(ceilf(sigma)));

    if (true == EnsureArenaMemory(width, height, quality, radius))
    {
        int proc_width = (quality == RenderQuality::Fast_HalfSize) ? (width >> 1) : width;
        int proc_height = (quality == RenderQuality::Fast_HalfSize) ? (height >> 1) : height;

        uint8_t* superBuffer = reinterpret_cast<uint8_t*>(g_arenaMem);
        size_t frameSize = static_cast<size_t>(proc_width) * static_cast<size_t>(proc_height);
        size_t alignedFloatSize = AlignSizeCuda(frameSize * sizeof(float));

        const int max_local_edges = (2 * radius + 1) * 2;
        const size_t max_edges = frameSize * max_local_edges;
        size_t alignedEdgeIdxSize = AlignSizeCuda(max_edges * sizeof(int));

        size_t offset = 0;
        float* d_Y = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_U = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_V = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;

        float* d_tA = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tB = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tC = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;

        float* d_tmpA = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tmpB = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tmpC = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;

        float* d_tAsm = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tBsm = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_tCsm = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;

        float* d_imProc1 = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;
        float* d_imProc2 = reinterpret_cast<float*>(superBuffer + offset); offset += alignedFloatSize;

        int* d_I = reinterpret_cast<int*>(superBuffer + offset); offset += alignedEdgeIdxSize;
        int* d_J = reinterpret_cast<int*>(superBuffer + offset); offset += alignedEdgeIdxSize;

        dim3 blockDim(32, 16, 1);
        dim3 gridDimProc((proc_width + blockDim.x - 1) / blockDim.x, (proc_height + blockDim.y - 1) / blockDim.y, 1);
        dim3 gridDimFull((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

        // --- I/O & STRUCTURE TENSORS ---
        if (quality == RenderQuality::Fast_HalfSize) Kernel_AdobeToPlanar<true> << <gridDimProc, blockDim, 0, stream >> > (inBuffer, srcPitch, d_Y, d_U, d_V, proc_width, proc_height, width, height, isFloat16);
        else Kernel_AdobeToPlanar<false> << <gridDimFull, blockDim, 0, stream >> > (inBuffer, srcPitch, d_Y, d_U, d_V, width, height, width, height, isFloat16);

        Kernel_ComputeStructureTensors << <gridDimProc, blockDim, 0, stream >> > (d_Y, d_tA, d_tB, d_tC, proc_width, proc_height);

        // --- VECTORIZED BLUR ---
        Kernel_Blur_Horizontal_Vec << <gridDimProc, blockDim, 0, stream >> > (d_tA, d_tB, d_tC, d_tmpA, d_tmpB, d_tmpC, proc_width, proc_height, radius);
        Kernel_Blur_Vertical_Vec << <gridDimProc, blockDim, 0, stream >> > (d_tmpA, d_tmpB, d_tmpC, d_tAsm, d_tBsm, d_tCsm, proc_width, proc_height, radius);

        // --- FUSED GRAPH BUILDER ---
        float angle_tol = cosf(angular * (3.14159265f / 180.0f));
        float rot_rad = angle * (3.14159265f / 180.0f);

        Kernel_BuildEdgeGraph_Fused << <gridDimProc, blockDim, 0, stream >> > (d_tAsm, d_tBsm, d_tCsm, d_I, d_J, proc_width, proc_height, radius, max_local_edges, angle_tol * angle_tol, cosf(rot_rad), sinf(rot_rad));

        // --- TEMPLATED PIXEL-PARALLEL MORPHOLOGY ---
        cudaMemcpyAsync(d_imProc1, d_Y, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        dim3 blockPixels(256);
        dim3 gridPixels((static_cast<unsigned int>(frameSize) + blockPixels.x - 1) / blockPixels.x);

        for (int step = 0; step < iter; ++step)
        {
            if (bias == StrokeBias::DarkBias_Open)
            {
                cudaMemcpyAsync(d_imProc2, d_imProc1, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                Kernel_Morph<false> << <gridPixels, blockPixels, 0, stream >> > (d_imProc1, d_imProc2, d_I, d_J, frameSize, max_local_edges, bias_val);
                cudaMemcpyAsync(d_imProc1, d_imProc2, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                Kernel_Morph<true> << <gridPixels, blockPixels, 0, stream >> > (d_imProc2, d_imProc1, d_I, d_J, frameSize, max_local_edges, bias_val);
            }
            else if (bias == StrokeBias::LightBias_Close)
            {
                cudaMemcpyAsync(d_imProc2, d_imProc1, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                Kernel_Morph<true> << <gridPixels, blockPixels, 0, stream >> > (d_imProc1, d_imProc2, d_I, d_J, frameSize, max_local_edges, bias_val);
                cudaMemcpyAsync(d_imProc1, d_imProc2, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                Kernel_Morph<false> << <gridPixels, blockPixels, 0, stream >> > (d_imProc2, d_imProc1, d_I, d_J, frameSize, max_local_edges, bias_val);
            }
            else
            {
                cudaMemcpyAsync(d_imProc2, d_imProc1, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                if (step % 2 == 0) Kernel_Morph<false> << <gridPixels, blockPixels, 0, stream >> > (d_imProc1, d_imProc2, d_I, d_J, frameSize, max_local_edges, bias_val);
                else Kernel_Morph<true> << <gridPixels, blockPixels, 0, stream >> > (d_imProc1, d_imProc2, d_I, d_J, frameSize, max_local_edges, bias_val);

                cudaMemcpyAsync(d_imProc1, d_imProc2, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                if (step % 2 == 0) Kernel_Morph<true> << <gridPixels, blockPixels, 0, stream >> > (d_imProc2, d_imProc1, d_I, d_J, frameSize, max_local_edges, bias_val);
                else Kernel_Morph<false> << <gridPixels, blockPixels, 0, stream >> > (d_imProc2, d_imProc1, d_I, d_J, frameSize, max_local_edges, bias_val);
            }
        }
        cudaMemcpyAsync(d_Y, d_imProc1, frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        if (quality == RenderQuality::Fast_HalfSize) Kernel_PlanarToAdobe<true> << <gridDimFull, blockDim, 0, stream >> > (inBuffer, srcPitch, outBuffer, dstPitch, d_Y, d_U, d_V, width, height, proc_width, proc_height, isFloat16);
        else Kernel_PlanarToAdobe<false> << <gridDimFull, blockDim, 0, stream >> > (inBuffer, srcPitch, outBuffer, dstPitch, d_Y, d_U, d_V, width, height, width, height, isFloat16);
    }

    cudaDeviceSynchronize();
}