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


#endif // __IMAGE_LAB_ART_POINTILISM_PALETTE_DEFINITION__