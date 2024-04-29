#include "ColorTemperatureControlsPresets.hpp"

bool setPresetsVector(std::vector<IPreset*>& v_presets) noexcept
{
	constexpr size_t mumberOfPresets{ 9 };
	v_presets.resize(mumberOfPresets);

	v_presets[0] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetLandscape);
	v_presets[1] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetNature);
	v_presets[2] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetMacro);
	v_presets[3] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetStreet);
	v_presets[4] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetPortraits);
	v_presets[5] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetNudeBody);
	v_presets[6] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetFood);
	v_presets[7] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetPainting);
	v_presets[8] = reinterpret_cast<IPreset*>(new (std::nothrow) PresetNightAndAstro);

	return true;
}

