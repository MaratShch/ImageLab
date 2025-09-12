#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_ENTRY__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_ENTRY__

#include <type_traits>
#include <array>

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
struct PEntry
{
    T r;
    T g;
    T b;
};


#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_ENTRY__