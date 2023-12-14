#ifndef __NOISE_CLEAN_ANISOTROPIC_DIFFUSION_GFUNCTION__
#define __NOISE_CLEAN_ANISOTROPIC_DIFFUSION_GFUNCTION__

#include <type_traits>

template <typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value || std::is_integral<T>::value, T>::type Gfunction (const T& x, const T& k) noexcept
{
	const T div = { x / k };
	constexpr T one = { 1 };
#ifdef _DEBUG
	const T divider = one + div * div;
	const T gResult = one / divider;
	return gResult;
#else
	return (one / (one + div * div));
#endif
}


#endif /* __NOISE_CLEAN_ANISOTROPIC_DIFFUSION_GFUNCTION__ */