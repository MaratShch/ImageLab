#include "ColorTemperatureControlsPresets.hpp"

bool setPresetsVector(std::vector<IPreset*>& v_presets)
{
	constexpr size_t mumberOfPresets{ 9 };
	v_presets.resize(mumberOfPresets);

	v_presets[0] = reinterpret_cast<IPreset*>(new PresetLandscape);
	v_presets[1] = reinterpret_cast<IPreset*>(new PresetNature);
	v_presets[2] = reinterpret_cast<IPreset*>(new PresetMacro);
	v_presets[3] = reinterpret_cast<IPreset*>(new PresetStreet);
	v_presets[4] = reinterpret_cast<IPreset*>(new PresetPortraits);
	v_presets[5] = reinterpret_cast<IPreset*>(new PresetNudeBody);
	v_presets[6] = reinterpret_cast<IPreset*>(new PresetFood);
	v_presets[7] = reinterpret_cast<IPreset*>(new PresetPainting);
	v_presets[8] = reinterpret_cast<IPreset*>(new PresetNightAndAstro);

	return true;
}

