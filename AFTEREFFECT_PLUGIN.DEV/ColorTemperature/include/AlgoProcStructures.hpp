#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__

#include <type_traits>

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
struct PixComponentsStr
{
    T   Y; // luma value from XYZ color space, represent Luma component
    T   u; // chromaticity coordinate from u'v' color space, represent green-red axes
    T   v; // chromaticity coordinate from u'v' color space, represent blue-yellow axes
};

using PixComponentsStr32  = PixComponentsStr<float>;
using PixComponentsStr64  = PixComponentsStr<double>;
using PixComponentsStr64l = PixComponentsStr<long double>;




#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__