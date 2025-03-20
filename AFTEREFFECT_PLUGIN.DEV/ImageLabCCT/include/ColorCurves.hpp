#ifndef __IMAGE_LAB_IMAGE_COLOR_CURVES_ALGORITHM__
#define __IMAGE_LAB_IMAGE_COLOR_CURVES_ALGORITHM__

#include <vector>
#include <cmath>

constexpr int32_t CURVES_X = 0;
constexpr int32_t CURVES_Y = 1;
constexpr int32_t CURVES_Z = 2;
constexpr int32_t CURVES_ALL = 3;

/* ========= COMPUTATION FOR 2 DEGREES 1931 OBSERVER ================== */
template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T x_sample_1931 (const T& waveLength) noexcept
{
	constexpr T threshold1{ static_cast<T>(442.0) };
	constexpr T threshold2{ static_cast<T>(599.8) };
	constexpr T threshold3{ static_cast<T>(501.1) };

	const T t1 = (waveLength - threshold1) * ((waveLength < threshold1) ? static_cast<T>(0.0624) : static_cast<T>(0.0374));
	const T t2 = (waveLength - threshold2) * ((waveLength < threshold2) ? static_cast<T>(0.0264) : static_cast<T>(0.0323));
	const T t3 = (waveLength - threshold3) * ((waveLength < threshold3) ? static_cast<T>(0.0490) : static_cast<T>(0.0382));
	
	constexpr T tHalf = static_cast<T>(-0.50);
	return (static_cast<T>(0.362) * std::exp(tHalf * t1 * t1) + 
		    static_cast<T>(1.056) * std::exp(tHalf * t2 * t2) -
		    static_cast<T>(0.065) * std::exp(tHalf * t3 * t3));
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T y_sample_1931 (const T& waveLength) noexcept
{
	constexpr T threshold1{ static_cast<T>(568.8) };
	constexpr T threshold2{ static_cast<T>(530.9) };

	const T t1 = (waveLength - threshold1) * ((waveLength < threshold1) ? static_cast<T>(0.02130) : static_cast<T>(0.02470));
	const T t2 = (waveLength - threshold2) * ((waveLength < threshold2) ? static_cast<T>(0.06130) : static_cast<T>(0.03220));
	
	constexpr T tHalf = static_cast<T>(-0.50);
	return (static_cast<T>(0.821) * std::exp(tHalf * t1 * t1) + static_cast<T>(0.286) * std::exp(tHalf * t2 * t2));
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T z_sample_1931 (const T& waveLength) noexcept
{
	constexpr T threshold1{ static_cast<T>(437.0) };
	constexpr T threshold2{ static_cast<T>(459.0) };

	const T t1 = (waveLength - threshold1) * ((waveLength < threshold1) ? static_cast<T>(0.08450) : static_cast<T>(0.02780));
	const T t2 = (waveLength - threshold2) * ((waveLength < threshold2) ? static_cast<T>(0.03850) : static_cast<T>(0.07250));

	constexpr T tHalf = static_cast<T>(-0.50);
	return (static_cast<T>(1.2170) * std::exp(tHalf * t1 * t1) + static_cast<T>(0.6810) * std::exp(tHalf * t2 * t2));
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::vector<std::vector<T>> generate_color_curves_1931_observer (const T& minWlength, const T& maxWlength, const T& step) noexcept
{
    const size_t vectorSize = static_cast<size_t>((maxWlength - minWlength) / step) + 1u;
    std::vector<std::vector<T>> vectorCurves(vectorSize, std::vector<T>(CURVES_ALL));

    /* generate X, Y, Z color curves for CIE-1931 2 degrees observer */
    T WaveLength{ minWlength };
	for (size_t i = 0; i < vectorSize; i++)
	{
		/* this computation happened on initialization stage, so we may use std::exp() functions for compute point curves value */
		vectorCurves[i][CURVES_X] = x_sample_1931(WaveLength);
		vectorCurves[i][CURVES_Y] = y_sample_1931(WaveLength);
		vectorCurves[i][CURVES_Z] = z_sample_1931(WaveLength);
		WaveLength += step;
	}
	
	return vectorCurves;
}


/* ========= COMPUTATION FOR 10 DEGREES 1964 OBSERVER ================= */
template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T x_sample_1964(const T& waveLength) noexcept
{
	const T a  = std::log((waveLength + static_cast<T>(570.0)) / static_cast<T>(1014.0));
	const T t1 = static_cast<T>(0.40) * std::exp(static_cast<T>(-1250.0) * a * a);
	const T b  = std::log((static_cast<T>(1338.0) - waveLength) / static_cast<T>(743.50));
	const T t2 = static_cast<T>(1.130) * std::exp(static_cast<T>(-234.0) * b * b);
	return (t1 + t2);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T y_sample_1964(const T& waveLength) noexcept
{
	const T t = (waveLength - static_cast<T>(556.10)) / static_cast<T>(46.140);
	return (static_cast<T>(1.0110) * std::exp(static_cast<T>(-0.50) * t * t));
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline T z_sample_1964(const T& waveLength) noexcept
{
	const T t = std::log((waveLength - static_cast<T>(266.0)) / static_cast<T>(180.40));
	return (static_cast<T>(2.060) * std::exp(static_cast<T>(-32.0) * t * t));
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::vector<std::vector<T>> generate_color_curves_1964_observer (const T& minWlength, const T& maxWlength, const T& step) noexcept
{
    const size_t vectorSize = static_cast<size_t>((maxWlength - minWlength) / step) + 1u;
    std::vector<std::vector<T>> vectorCurves(vectorSize, std::vector<T>(CURVES_ALL));

    /* generate X, Y, Z color curve for CIE-1964 10 degrees observer */
	T WaveLength{ minWlength };
	for (size_t i = 0; i < vectorSize; i++)
	{
		/* this computation happened on initialization stage, so we may use std::exp() functions for compute point curves value */
		vectorCurves[i][CURVES_X] = x_sample_1964(WaveLength);
		vectorCurves[i][CURVES_Y] = y_sample_1964(WaveLength);
		vectorCurves[i][CURVES_Z] = z_sample_1964(WaveLength);
		WaveLength += step;
	}

	return vectorCurves;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::tuple<const T, const T, const T> compute_scalar_XYZ(const std::vector<std::vector<T>>& observer, const std::vector<T>& illuminant) noexcept
{
    T scalarX{ static_cast<T>(0) };
    T scalarY{ static_cast<T>(0) };
    T scalarZ{ static_cast<T>(0) };

    const auto size = illuminant.size();
    if (observer.size() == size)
    {
        for (size_t i = 0; i < size; i++)
        {
            scalarX += (observer[i][CURVES_X] * illuminant[i]);
            scalarY += (observer[i][CURVES_Y] * illuminant[i]);
            scalarZ += (observer[i][CURVES_Z] * illuminant[i]);
        }
    }

    return std::make_tuple(scalarX, scalarY, scalarZ);
}


#endif /* __IMAGE_LAB_IMAGE_COLOR_CURVES_ALGORITHM__ */