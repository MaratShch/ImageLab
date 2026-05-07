#include <cstring> // For std::memcpy
#include <cstdint>

void ApplyMirroredPadding
(
    float* inY, 
    const int32_t width, 
    const int32_t height, 
    const int32_t stride, 
    const int32_t radius // The currently requested UI radius
)
{
    if (radius <= 0) return;

    // ==========================================
    // STEP 1: Pad Top and Bottom (Image width only)
    // ==========================================
    for (int32_t r = 1; r <= radius; ++r) 
    {
        // Source rows (moving inwards from the edge)
        const float* srcTop = inY + (r - 1) * stride;
        const float* srcBottom = inY + (height - r) * stride;

        // Destination rows (moving outwards into the halo)
        // Notice the negative stride for dstTop!
        float* dstTop = inY - (r * stride);
        float* dstBottom = inY + (height + r - 1) * stride;

        // Fast memory copy for the horizontal rows
        size_t rowBytes = width * sizeof(float);
        std::memcpy(dstTop, srcTop, rowBytes);
        std::memcpy(dstBottom, srcBottom, rowBytes);
    }

    // ==========================================
    // STEP 2: Pad Left and Right (Including the corners)
    // ==========================================
    // TRICK: We start 'y' at -radius and end at height + radius. 
    // This loops over the actual image AND the top/bottom padding we just created,
    // which automatically mirrors the corners diagonally!
    
    int32_t startY = -radius;
    int32_t endY = height + radius;

    for (int32_t y = startY; y < endY; ++y) 
    {
        // Get the pointer to the absolute start of the current row
        float* rowPtr = inY + (y * stride);

        // Mirror the pixels outwards to the left and right
        for (int32_t r = 1; r <= radius; ++r) 
        {
            // Left mirror (negative index)
            rowPtr[-r] = rowPtr[r - 1];
            
            // Right mirror (past the width)
            rowPtr[width + r - 1] = rowPtr[width - r];
        }
    }
	
	return;
}