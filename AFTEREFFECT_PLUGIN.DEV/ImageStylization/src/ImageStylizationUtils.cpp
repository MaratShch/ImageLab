#include "ImageStylization.hpp"

CACHE_ALIGN static float RandomValBuffer[RandomBufSize]{};


uint32_t utils_get_random_value(void)
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


void utils_generate_random_values (float* pBuffer, const uint32_t& bufSize)
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

void utils_create_random_buffer(void)
{
	static_assert(IsPowerOf2(RandomBufSize), "Random buffer size isn't power of 2");
	utils_generate_random_values(RandomValBuffer, RandomBufSize);
	return;
}


const float* __restrict get_random_buffer (uint32_t& size)
{
	size = RandomBufSize;
	return RandomValBuffer;
}

const float* __restrict get_random_buffer (void)
{
	return RandomValBuffer;
}