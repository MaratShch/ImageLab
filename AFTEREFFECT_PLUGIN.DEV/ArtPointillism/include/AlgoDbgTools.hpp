#include "AlgoMemHandler.hpp"

inline void AlgoDbgColorReplace
(
	const MemHandler& algoMemHandler,
    int32_t width,
	int32_t height
) noexcept
{
	// [DEBUG] OVERRIDE INPUT WITH SOLID RED
	const int32_t num_pixels = width * height;
	for (int i = 0; i < num_pixels; ++i)
	{
		// Override L, a, b buffers to be "Red"
		// Red in Lab is approx: L=53, a=80, b=67
		algoMemHandler.L[i] = 53.0f;
		algoMemHandler.ab[i*2+0] = 80.0f; 
		algoMemHandler.ab[i*2+1] = 67.0f;
	}

	return;
}
