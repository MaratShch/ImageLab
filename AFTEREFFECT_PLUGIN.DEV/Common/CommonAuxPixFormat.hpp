#pragma once

#include <cfloat>

#pragma pack(push)
#pragma pack(1)

template <typename T>
struct _tCIELabPix
{
	T L;
	T a;
	T b;
};

using fCIELabPix   = _tCIELabPix<float>;
using dCIELabPix   = _tCIELabPix<double>;
using i32CIELabPix = _tCIELabPix<int32_t>;

constexpr size_t fCIELabPix_size = sizeof(fCIELabPix);
constexpr size_t dCIELabPix_size = sizeof(dCIELabPix);
constexpr size_t i32CIELabPix_size = sizeof(i32CIELabPix);


template <typename T>
struct _tRGB
{
	T R;
	T G;
	T B;
};

using fRGB = _tRGB<float>;
constexpr size_t fRGB_size = sizeof(fRGB);

#pragma pack(pop)
