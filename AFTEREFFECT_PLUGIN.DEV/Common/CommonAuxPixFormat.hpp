#ifndef __IMAGE_LAB_AUXILIARY_PIXEL_FORMAT_INTERNAL__
#define __IMAGE_LAB_AUXILIARY_PIXEL_FORMAT_INTERNAL__

/* define AUX pixel format, using internally by image processing API's */
#include <cfloat>
#include <cstdint>

#pragma pack(push)
#pragma pack(1)

template <typename T>
struct _tXYZPix
{
	T X;
	T Y;
	T Z;
};

using fXYZPix   = _tXYZPix<float>;
using dXYZPix   = _tXYZPix<double>;
using i32XYZPix = _tXYZPix<int32_t>;

constexpr size_t fXYZPix_size   = sizeof(fXYZPix);
constexpr size_t dXYZPix_size   = sizeof(dXYZPix);
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


template <typename T>
struct _tYUV /* YUV pixel value range */
{
	T Y;
	T U;
	T V;
};

using fYUV = _tYUV<float>;
using dYUV = _tYUV<double>;

constexpr size_t fYUV_size = sizeof(fYUV);
constexpr size_t dYUV_size = sizeof(dYUV);


#pragma pack(pop)

#endif /* __IMAGE_LAB_AUXILIARY_PIXEL_FORMAT_INTERNAL__ */