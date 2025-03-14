#ifndef __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__
#define __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__

#include "AlgoPlanckianLocus.hpp"
#include <limits>

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant (const T& minWlength, const T& maxWlength, const T& step, const T& whitePoint) noexcept
{
	const size_t vectorSize = static_cast<size_t>((maxWlength - minWlength) / step) + 1;
	std::vector<T> spectralRadiance(vectorSize);
	T maxIlluminant {std::numeric_limits<T>::min()};
	T waveLength {minWlength};

	/* compute non normalized Spectral Radiance values in correspondent to defined wave length */
	for (size_t i = 0u; i < vectorSize; i++)
	{
		T value = PlanckianLocus::Planck (waveLength, whitePoint);
		maxIlluminant = std::max(value, maxIlluminant);
		spectralRadiance[i] = value;
		waveLength += step;
	}

	/* Normalize computed Spectral Radiance values before return to caller. */
	/* Perform in-place normalizing											*/					
	PlanckianLocus::SpectralRadianceNormalize (maxIlluminant, spectralRadiance);

	return spectralRadiance;
}

#endif /* __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__ */