#pragma once

#include <cfloat>

#pragma pack(push)
#pragma pack(1)

template <typename T>
struct _tXYZPix
{
	T X;
	T Y;
	T Z;
};

using fXYZPix    = _tXYZPix<float>;
using dXYZLabPix = _tXYZPix<double>;
using i32XYZPix  = _tXYZPix<int32_t>;

constexpr size_t fXYZPix_size   = sizeof(fXYZPix);
constexpr size_t dXYZPix_size   = sizeof(dXYZLabPix);
constexpr size_t i32XYZPix_size = sizeof(i32XYZPix);


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
struct _tRGB /* sRGB pixel value range */
{
	T R;
	T G;
	T B;
};

using fRGB = _tRGB<float>;
using dRGB = _tRGB<double>;

constexpr size_t fRGB_size = sizeof(fRGB);
constexpr size_t dRGB_size = sizeof(dRGB);

#pragma pack(pop)
