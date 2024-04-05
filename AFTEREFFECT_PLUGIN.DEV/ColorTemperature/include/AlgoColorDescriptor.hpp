#ifndef __IMAGE_LAB_ALGORITHM_COLOR_DESCRUPTOR__
#define __IMAGE_LAB_ALGORITHM_COLOR_DESCRUPTOR__

#include <cmath>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include "ClassRestrictions.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorCurves.hpp"

using ColorDescriptorT = double;

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
class CAlgoColorDescriptor final
{
public:

	CLASS_NON_COPYABLE(CAlgoColorDescriptor);
	CLASS_NON_MOVABLE (CAlgoColorDescriptor);

	static CAlgoColorDescriptor<T>* getInstance (void)
	{
		CAlgoColorDescriptor<T>* iDescriptor = s_instance.load(std::memory_order_acquire);
		if (nullptr == iDescriptor)
		{
			std::lock_guard<std::mutex> myLock(s_protectMutex);
			iDescriptor = s_instance.load(std::memory_order_relaxed);
			if (nullptr == iDescriptor)
			{
				iDescriptor = new CAlgoColorDescriptor<T>();
				s_instance.store(iDescriptor, std::memory_order_release);
			}
		}
		return iDescriptor;
	} /* static MemoryInterface* getInstance() */


	bool Initialize (void) noexcept
	{
		if (false == m_IsInitialized)
		{
			generateObservers();
			generateIlluminant();
			m_IsInitialized = true;
		}
		return true;
	}

	std::vector<std::vector<T>>& getObserver (bool is1931) const noexcept { return (true == is1931 ? m_Cmf1931 : m_Cmf1964); }
	std::vector<T>& getIlluminant (const eIlluminant& illuminant) const noexcept { return m_Illuminant[illuminant]; }

protected:
private:
	static std::atomic<CAlgoColorDescriptor<T>*> s_instance;
	static std::mutex s_protectMutex;

	std::atomic<bool> m_IsInitialized = false;

	const T m_waveLengthStart = waveLengthStart;
	const T m_waveLengthStop  = waveLengthStop;
	const T m_wavelengthStep  = wavelengthStepFinest;

	const T m_WhitePoint_D65 = 6504.0;
	const T m_WhitePoint_D65_Cloudy_Tint = 0.030;
	const T m_Whitepoint_Tungsten = 3200.0;
	const T m_WhitePoint_FluorescentDayLight = 6500.0;
	const T m_WhitePoint_FluorescentWarmWhite = 3000.0;
	const T m_WhitePoint_FluorescentSoftWhite = 4200.0;
	const T m_WhitePoint_Incandescent = 2700.0;
	const T m_WhitePoint_Moonlight = 4100.0;

	std::vector<std::vector<T>> m_Cmf1931;
	std::vector<std::vector<T>> m_Cmf1964;

	std::vector<std::vector<T>> m_Illuminant;

	CAlgoColorDescriptor () = default;
	~CAlgoColorDescriptor() = default;

	void generateObservers (void) noexcept
	{
		/* Initialize Color Matching Functions (CMF) for: "2 degrees 1931" and "10 degrees 1964" observers */
		m_Cmf1931 = generate_color_curves_1931_observer (m_waveLengthStart, m_waveLengthStop, m_wavelengthStep);
		m_Cmf1964 = generate_color_curves_1964_observer (m_waveLengthStart, m_waveLengthStop, m_wavelengthStep);
		return;
	}

	void generateIlluminant (void) noexcept
	{
		m_Illuminant.resize(static_cast<size_t>(COLOR_TEMPERATURE_TOTAL_ILLUMINANTS));
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_D65]               = init_illuminant_D65<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_D65_CLOUDY]        = init_illuminant_D65_Cloudy<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_TUNGSTEN]          = init_illuminant_Tungsten<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_FLUORESCENT]       = init_illuminant_FluorescentDayLight<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_WHITE_FLUORESCENT] = init_illuminant_WhiteFluorescent<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_INCANDESCENT]      = init_illuminant_Incandescent<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_WARM_WHITE]        = init_illuminant_FluorescentWarmWhite<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_SOFT_WHITE]        = init_illuminant_FluorescentSoftWhite<T>();
		m_Illuminant[COLOR_TEMPERATURE_ILLUMINANT_MOONLIGHT]         = init_illuminant_Moonlight<T>();
		return;
	}

}; /* class CAlgoColorDescriptor */


#endif /* __IMAGE_LAB_ALGORITHM_COLOR_DESCRUPTOR__ */