#include "BlackAndWhite.hpp"
#include "ColorTransformMatrix.hpp"

template <typename U, typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ProcessImage
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              src_linePitch,
	A_long              dst_linePitch,
	const U             noColor
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_linePitch;
		      T* __restrict pDstLine = pDst + j * dst_linePitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].V = noColor;
			pDstLine[i].U = noColor;
			pDstLine[i].Y = pSrcLine[i].Y;
			pDstLine[i].A = pSrcLine[i].A;
		}
	}
	return;
}


template <typename U, typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ProcessImage
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              src_linePitch,
	A_long              dst_linePitch,
	const U             noColor
) noexcept
{
	float constexpr colorMatrix1[3] = { RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2] };

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_linePitch;
		      T* __restrict pDstLine = pDst + j * dst_linePitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = pSrcLine[i].R * colorMatrix1[0] + pSrcLine[i].G * colorMatrix1[1] + pSrcLine[i].B * colorMatrix1[2];
			pDstLine[i].A = pSrcLine[i].A;
		}
	}
	return;
}


template <typename U, typename T, std::enable_if_t<is_no_alpha_channel<T>::value>* = nullptr>
inline void ProcessImage
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              src_linePitch,
	A_long              dst_linePitch,
	const U             noColor
) noexcept
{
	float constexpr colorMatrix1[3] = { RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2] };

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_linePitch;
		      T* __restrict pDstLine = pDst + j * dst_linePitch;
		for (A_long i = 0; i < sizeX; i++)
			pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = pSrcLine[i].R * colorMatrix1[0] + pSrcLine[i].G * colorMatrix1[1] + pSrcLine[i].B * colorMatrix1[2];
	}
	return;
}
