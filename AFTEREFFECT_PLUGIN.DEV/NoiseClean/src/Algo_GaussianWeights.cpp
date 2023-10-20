#include "NoiseClean.hpp"
#define FAST_COMPUTE_EXTRA_PRECISION
#include "FastAriphmetics.hpp"


void gaussian_weights (A_long filterRadius, float gMesh[][cBilateralWindowMax]) noexcept
{
	A_long i, j;
	A_long x, y;

	constexpr float divider = 2.0f * cBilateralGaussSigma * cBilateralGaussSigma;
	for (y = -filterRadius, j = 0; j <= filterRadius; j++, y++)
	{
		__LOOP_UNROLL(3)
		for (x = -filterRadius, i = 0; i <= filterRadius; i++, x++)
		{
			const float dSum = static_cast<float>((x * x) + (y * y));
			gMesh[j][i] = FastCompute::Exp(-dSum / divider);
		}
	}

	return;
}


void gaussian_weights (A_long filterRadius, float* __restrict gMesh) noexcept
{
	A_long j, x, y;

	constexpr float divider_reciproc = 1.0f / (2.0f * cBilateralGaussSigma * cBilateralGaussSigma);
	const A_long filterDim = filterRadius * filterRadius;

	for (x = y = -filterRadius, j = 0; j < filterDim; j++, x++)
	{
		if (x > filterRadius)
		{
			x = -filterRadius;
			y++;
		}

		const float dSum = static_cast<float>((x * x) + (y * y));
		*gMesh++ = FastCompute::Exp(-dSum * divider_reciproc);
	}

	return;
}