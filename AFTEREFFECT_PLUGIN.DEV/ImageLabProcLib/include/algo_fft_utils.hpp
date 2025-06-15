#ifndef __IMAGE_LAB_ALGORITHM_UTILS_FOR_FFT__
#define __IMAGE_LAB_ALGORITHM_UTILS_FOR_FFT__

#include <cstdint>
#include "FastAriphmetics.hpp"

constexpr int32_t maxPaddingSize = 8192;

inline uint32_t get_next_power_of_2 (uint32_t n) noexcept
{
    if (n == 0)
        return 1;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

inline int32_t get_next_power_of_2 (int32_t n) noexcept
{
    if (n == 0)
        return 1;

    uint32_t un = static_cast<uint32_t>(n);
    return static_cast<int32_t>(get_next_power_of_2(un));
}


inline void get_padded_image_size (const uint32_t& sizeX, const uint32_t& sizeY, uint32_t& paddedX, uint32_t& paddedY) noexcept
{
    paddedX = FastCompute::Min(static_cast<uint32_t>(maxPaddingSize), get_next_power_of_2(sizeX));
    paddedY = FastCompute::Min(static_cast<uint32_t>(maxPaddingSize), get_next_power_of_2(sizeY));
    return;
}


inline void get_padded_image_size(const int32_t& sizeX, const int32_t& sizeY, int32_t& paddedX, int32_t& paddedY) noexcept
{
    paddedX = FastCompute::Min(maxPaddingSize, get_next_power_of_2(sizeX));
    paddedY = FastCompute::Min(maxPaddingSize, get_next_power_of_2(sizeY));
    return;
}


inline void get_padded_image_size_x2 (const uint32_t& sizeX, const uint32_t& sizeY, uint32_t& paddedX, uint32_t& paddedY) noexcept
{
    paddedX = sizeX * 2u;
    paddedY = sizeY * 2u;
    return;
}


inline void get_padded_image_size_x2 (const int32_t& sizeX, const int32_t& sizeY, int32_t& paddedX, int32_t& paddedY) noexcept
{
    paddedX = sizeX * 2;
    paddedY = sizeY * 2;
    return;
}


#endif // __IMAGE_LAB_ALGORITHM_UTILS_FOR_FFT__