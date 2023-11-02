#include "BlackAndWhite.hpp"
#include "FastAriphmetics.hpp"
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


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ProcessImageAdvanced
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              src_linePitch,
	A_long              dst_linePitch
) noexcept
{
	constexpr float fCoeff[3] = { 0.2126f,   0.7152f,  0.0722f };
	constexpr float fLumaExp = 1.0f / 2.20f;

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_linePitch;
		      T* __restrict pDstLine = pDst + j * dst_linePitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			const float fTmpVal =
				fCoeff[0] * FastCompute::Pow(static_cast<float>(pSrcLine[i].R), 2.20f) +
				fCoeff[1] * FastCompute::Pow(static_cast<float>(pSrcLine[i].G), 2.20f) +
				fCoeff[2] * FastCompute::Pow(static_cast<float>(pSrcLine[i].B), 2.20f);

			pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = FastCompute::Pow (fTmpVal, fLumaExp);
			pDstLine[i].A = pSrcLine[i].A;
		}
	}

	return;
}

template <typename T, std::enable_if_t<is_no_alpha_channel<T>::value>* = nullptr>
inline void ProcessImageAdvanced
(
	const T* __restrict pSrc,
	T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              src_linePitch,
	A_long              dst_linePitch
) noexcept
{
	constexpr float fCoeff[3] = { 0.2126f,   0.7152f,  0.0722f };
	constexpr float fLumaExp = 1.0f / 2.20f;

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_linePitch;
		      T* __restrict pDstLine = pDst + j * dst_linePitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			const float fTmpVal =
				fCoeff[0] * FastCompute::Pow(static_cast<float>(pSrcLine[i].R), 2.20f) +
				fCoeff[1] * FastCompute::Pow(static_cast<float>(pSrcLine[i].G), 2.20f) +
				fCoeff[2] * FastCompute::Pow(static_cast<float>(pSrcLine[i].B), 2.20f);

			pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = FastCompute::Pow(fTmpVal, fLumaExp);
		}
	}

	return;
}