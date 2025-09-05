#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_PALETTE__
#define __IMAGE_LAB_RETRO_VISION_FILTER_PALETTE__

#include <type_traits>
#include "PaletteCGA.hpp"
#include "PaletteEGA.hpp"
#include "PaletteVGA.hpp"

template <typename T>
class is_RETRO_PALETTE
{
    /**
    * RGB proc variation
    */
    template <typename TT,
        typename std::enable_if<
        std::is_same<TT, CGA_Palette>::value      ||
        std::is_same<TT, CGA_PaletteF32>::value   ||
        std::is_same<TT, EGA_Palette>::value      ||
        std::is_same<TT, EGA_PaletteF32>::value   ||
        std::is_same<TT, VGA_Palette16>::value    ||
        std::is_same<TT, VGA_Palette256>::value   ||
        std::is_same<TT, VGA_Palette16F32>::value ||
        std::is_same<TT, VGA_Palette256F32>::value>::type* = nullptr>
        static auto test(int)->std::true_type;

    template<typename>
    static auto test(...)->std::false_type;

public:
    static constexpr const bool value = decltype(test<T>(0))::value;
};



#endif // __IMAGE_LAB_RETRO_VISION_FILTER_PALETTE__