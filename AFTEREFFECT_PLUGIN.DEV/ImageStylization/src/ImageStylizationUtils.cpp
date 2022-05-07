#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "FastAriphmetics.hpp"


CACHE_ALIGN static float RandomValBuffer[RandomBufSize]{};

//data structure to store the cost of merging intervals of the histogram
typedef struct costdata {
	double cost;
	int imin1, imin2;
	int typemerging;
};

uint32_t utils_get_random_value(void) noexcept
{
	// used xorshift algorithm
	static uint32_t x = 123456789u;
	static uint32_t y = 362436069u;
	static uint32_t z = 521288629u;
	static uint32_t w = 88675123u;
	static uint32_t t;

	t = x ^ (x << 11);
	x = y; y = z; z = w;
	return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}


void utils_generate_random_values (float* pBuffer, const uint32_t& bufSize) noexcept
{
	constexpr float fLimit = static_cast<float>(UINT_MAX);
	if (nullptr != pBuffer && 0u != bufSize)
	{
		for (uint32_t idx = 0u; idx < bufSize; idx++)
		{
			pBuffer[idx] = static_cast<float>(utils_get_random_value()) / fLimit;
		}
	}
	return;
}

void utils_create_random_buffer(void) noexcept
{
	static_assert(IsPowerOf2(RandomBufSize), "Random buffer size isn't power of 2");
	utils_generate_random_values(RandomValBuffer, RandomBufSize);
	return;
}


const float* __restrict get_random_buffer (uint32_t& size) noexcept
{
	size = RandomBufSize;
	return RandomValBuffer;
}

const float* __restrict get_random_buffer (void) noexcept 
{
	return RandomValBuffer;
}




/* =================================================================== */
/* simple and rapid (non precise) convert RGB to Black-abd-White image */
/* =================================================================== */
template <class T, class U, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
void Rgb2Bw
(
	const T* __restrict pSrc,
	U* __restrict pDst,
	const A_long&       width,
	const A_long&       height,
	const A_long&       pitch
)
{
	for (A_long j = 0; j < height; j++)
	{
		for (A_long i = 0; i < width; i++)
		{
			const A_long idx{ j * pitch + i };
			pDst[idx] = (static_cast<int>(pSrc[idx].R) + static_cast<int>(pSrc[idx].G) + static_cast<int>(pSrc[idx].B)) / 3;
		}
	}
	return;
}

void Rgb2Bw
(
	const PF_Pixel_BGRA_32f* __restrict pSrc,
	PF_FpShort*        __restrict pDst,
	const A_long&       width,
	const A_long&       height,
	const A_long&       pitch
)
{
	constexpr float reciproc3{ 1.f / 3.f };
	for (A_long j = 0; j < height; j++)
	{
		for (A_long i = 0; i < width; i++)
		{
			const A_long idx{ j * pitch + i };
			pDst[idx] = (pSrc[idx].R + pSrc[idx].G + pSrc[idx].B) * reciproc3;
		}
	}
	return;
}

