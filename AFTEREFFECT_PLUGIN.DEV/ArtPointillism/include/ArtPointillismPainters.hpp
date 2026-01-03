#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTERS_TECHNISC_SPECIFIC_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTERS_TECHNISC_SPECIFIC_DEFINITIONS__

#include "ArtPointillismCross.hpp"
#include "ArtPointillismLuce.hpp"
#include "ArtPointillismMatisse.hpp"
#include "ArtPointillismPissaro.hpp"
#include "ArtPointillismRysselberghe.hpp"
#include "ArtPointillismSeurat.hpp"
#include "ArtPointillismSignac.hpp"
#include "ArtPointillismVanGogh.hpp"


template <typename T>
class is_PAINTER_PALETTE_U8
{
    /**
    * PALETTE type
    */
    template <typename TT,
        typename std::enable_if<
        std::is_same<TT, Cross_Palette_u8>::value           ||
        std::is_same<TT, Luce_Palette_u8>::value            ||
        std::is_same<TT, Matisse_Palette_u8>::value         ||
        std::is_same<TT, Pissaro_Palette_u8>::value         ||
        std::is_same<TT, Rysselberghe_Palette_u8>::value    ||
        std::is_same<TT, Seurat_Palette_u8>::value          ||
        std::is_same<TT, Signac_Palette_u8>::value          ||
        std::is_same<TT, VanGogh_Palette_u8>::value>::type* = nullptr>
        static auto test(int)->std::true_type;

    template<typename>
    static auto test(...)->std::false_type;

public:
    static constexpr const bool value = decltype(test<T>(0))::value;
};


template <typename T>
class is_PAINTER_PALETTE_F32
{
    /**
    * PALETTE type
    */
    template <typename TT,
        typename std::enable_if<
        std::is_same<TT, Cross_Palette_f32>::value          ||
        std::is_same<TT, Luce_Palette_f32>::value           ||
        std::is_same<TT, Matisse_Palette_f32>::value        ||
        std::is_same<TT, Pissaro_Palette_f32>::value        ||
        std::is_same<TT, Rysselberghe_Palette_f32>::value   ||
        std::is_same<TT, Seurat_Palette_f32>::value         ||
        std::is_same<TT, Signac_Palette_f32>::value         ||
        std::is_same<TT, VanGogh_Palette_f32>::value>::type* = nullptr>
        static auto test(int)->std::true_type;

    template<typename>
    static auto test(...)->std::false_type;

public:
    static constexpr const bool value = decltype(test<T>(0))::value;
};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTERS_TECHNISC_SPECIFIC_DEFINITIONS__