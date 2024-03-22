#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_PRESET_CLASS__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_PRESET_CLASS__

#include "ColorTemperatureControls.hpp"
#include "ColorTemperatureEnums.hpp"

typedef struct {
	double Tx;
	double Ty;
} sTemperaturePoint;

class IPreset
{
public:
	virtual const eObservers		getObserver  (void) = 0;
	virtual const eIlluminant		getIlluminant(void) = 0;
	virtual const eWaveLenghthStep	getWavelengthStep(void) = 0;
	virtual const sTemperaturePoint getTemperaturePoint(void) = 0;
};

/* 
Preset for Landscape Photo/Video.
Observer: This observer provides a broader and more averaged view of color perception, suitable for capturing the wide range of colors in landscapes.
Illuminant: Adjusted for the slightly warmer tones often seen in cloudy conditions.
Step: A finer wavelength step allows for more precise capture of the varied colors in landscapes.
*/
class PresetLandscape final : public IPreset
{
public:
	const eObservers getObserver(void) {return m_observer;}
	const eIlluminant getIlluminant(void) {return m_illuminant;}
	const eWaveLenghthStep getWavelengthStep(void) {return m_step;}
	const sTemperaturePoint getTemperaturePoint(void) {return m_IlluminantTempPoint;}
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1964_10;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_D65_CLOUDY;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3333, 0.3333 };
};

/*
Preset for Nature Photo/Video
Observer: This observer provides a broader view of color perception for capturing the variety of colors in nature.
Illuminant: Standard reference for natural light.
Step: Standard wavelength step for nature scenes with diverse colors.
*/
class PresetNature final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1964_10;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_D65;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3127, 0.3290 };
};

/*
Preset for Macro Photo/Video.
Observer: This observer provides a broader view of color perception for capturing the variety of colors in nature.
Illuminant: Standard reference for natural light.
Step: A finest step for capturing the nuances in small objects.
*/
class PresetMacro final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1964_10;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_D65;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINEST;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3127, 0.3290 };
};

/*
Preset for Street Photo/Video.
Observer: Similar to landscape photography, this observer captures a wide range of colors typically found in urban environments.
Illuminant: Standard reference for daylight conditions.
Step: Suitable for the varied colors encountered in street scenes.
*/
class PresetStreet final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1964_10;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_D65;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3127, 0.3290 };
};

/*
Preset for Portraits Photo/Video.
Observer: This observer provides a more focused view of color perception, suitable for capturing the subtle nuances in skin tones and facial features.
Illuminant: Useful for adjusting colors under indoor tungsten lighting.
Step: A standard wavelength step for general portrait photography.
*/
class PresetPortraits final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1931_2;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_TUNGSTEN;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.4476, 0.4074 };
};

/*
Preset for Nude Photo/Video.
Observer: Similar to portrait photography, this observer captures detailed color perception.
Illuminant: Adjusted for softer tones in indoor lighting.
Step: Standard wavelength step for capturing the nuances in skin tones.
*/
class PresetNudeBody final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1931_2;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_SOFT_WHITE;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3457, 0.3585 };
};

/*
Preset for Food Photo/Video.
Observer: This observer provides detailed color perception for capturing the vibrant colors of food.
Illuminant: Mimics the warm tones often used in food displays.
Step: A finer wavelength step for more precise color representation in food images.
*/
class PresetFood final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1931_2;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_WARM_WHITE;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINEST;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.3805, 0.3769 };
};

/*
Preset for Painting Photo/Video.
Observer: This observer provides a standard view of color perception.
Illuminant: Suitable for adjusting colors under indoor tungsten lighting.
Step: Standard wavelength step for general painting photography.
*/
class PresetPainting final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1931_2;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_INCANDESCENT;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.4476, 0.4074 };
};

/*
Preset for Nigh and Astro Photo/Video.
Observer: This observer provides a broader view of color perception.
Illuminant: Simulated cool tones for night sky shots.
Step: A finer step for capturing the subtle colors in night scenes.
*/
class PresetNightAndAstro final : public IPreset
{
public:
	const eObservers getObserver(void) { return m_observer; }
	const eIlluminant getIlluminant(void) { return m_illuminant; }
	const eWaveLenghthStep getWavelengthStep(void) { return m_step; }
	const sTemperaturePoint getTemperaturePoint(void) { return m_IlluminantTempPoint; }
private:
	const eObservers  m_observer = COLOR_TEMPERATURE_OBSERVER_1964_10;
	const eIlluminant m_illuminant = COLOR_TEMPERATURE_ILLUMINANT_MOONLIGHT;
	const eWaveLenghthStep m_step = COLOR_TEMPERATURE_WAVELENGTH_STEP_FINEST;
	const sTemperaturePoint m_IlluminantTempPoint = { 0.2852, 0.2741 };
};


#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_PRESET_CLASS__ */