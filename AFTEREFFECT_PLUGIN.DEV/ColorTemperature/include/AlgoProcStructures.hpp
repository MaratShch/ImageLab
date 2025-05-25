#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__

#include <type_traits>
#include "cct_interface.hpp"

template<typename T>
struct PixComponentsStr
{
    T   Y; // luma value from XYZ color space, represent Luma component
    T   u; // chromaticity coordinate from u'v' color space, represent green-red axes
    T   v; // chromaticity coordinate from u'v' color space, represent blue-yellow axes
};

using PixComponentsStr32  = PixComponentsStr<float>;
using PixComponentsStr64  = PixComponentsStr<double>;
using PixComponentsStr64l = PixComponentsStr<long double>;


typedef struct pHandle
{
    AEGP_PluginID id;
    A_long valid;
    AlgoCCT::CctHandleF32* hndl;

    pHandle::pHandle() noexcept
    {
        id = valid = 0;
        hndl = nullptr;
    }

}pHandle;

constexpr std::size_t pHandleStrSize{ sizeof(pHandle) };

#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_STRUCTURES__