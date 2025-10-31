#ifndef __IMAGE_LAB_ART_POINTILISM_PALETTE_DEFINITION__
#define __IMAGE_LAB_ART_POINTILISM_PALETTE_DEFINITION__

#include <type_traits>

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
struct PEntry
{
    T r;
    T g;
    T b;
};

template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
inline constexpr float F32 (const T val) noexcept { return static_cast<float>(val) / 255.0f; }


#endif // __IMAGE_LAB_ART_POINTILISM_PALETTE_DEFINITION__