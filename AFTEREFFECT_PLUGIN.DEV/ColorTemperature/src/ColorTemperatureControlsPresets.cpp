#include "ColorTemperatureControlsPresets.hpp"


bool setPresetsVector(std::vector<IPreset*>& v_presets) noexcept
{
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


void resetPresets(std::vector<IPreset*>& v_presets) noexcept
{
    if (mumberOfPresets == v_presets.size())
    {
        delete reinterpret_cast<PresetLandscape*>(v_presets[0]);
        delete reinterpret_cast<PresetNature   *>(v_presets[1]);
        delete reinterpret_cast<PresetMacro    *>(v_presets[2]);
        delete reinterpret_cast<PresetStreet   *>(v_presets[3]);
        delete reinterpret_cast<PresetPortraits*>(v_presets[4]);
        delete reinterpret_cast<PresetNudeBody *>(v_presets[5]);
        delete reinterpret_cast<PresetFood     *>(v_presets[6]);
        delete reinterpret_cast<PresetPainting *>(v_presets[7]);
        delete reinterpret_cast<PresetNightAndAstro*>(v_presets[8]);
    }

    v_presets.clear();

    return;
}