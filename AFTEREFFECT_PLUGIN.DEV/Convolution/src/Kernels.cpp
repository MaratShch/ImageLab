#pragma once

#include "Common.hpp"
#include "Matrix.hpp"

#if 0
CACHE_ALIGN constexpr int Sharpen[9] = 
{
	-1, -1, -1,
	-1,  9, -1,
	-1, -1, -1
};

CACHE_ALIGN constexpr int Blurr[9] =
{
	1, 2, 1,
	2, 4, 2,
	1, 2, 1
};
#endif