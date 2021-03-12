#include "ColorCorrectionHSL.hpp"
#include <windows.h>
#include <Shlobj.h>
#include <chrono>
#include <sstream>
#include <fstream>

constexpr uint32_t gVersion = 1u;
constexpr uint32_t gSubVersion = 0u;


LPSTR desktop_directory (void)
{
	static char path[MAX_PATH + 1];
	if (SHGetSpecialFolderPath(HWND_DESKTOP, path, CSIDL_MYDOCUMENTS, FALSE)) return path;
	else return NULL;
}

bool SaveCustomSetting_1_0
(
	PF_ParamDef* params[]
) noexcept
{
	CACHE_ALIGN strHslRecord strParams{};
	CACHE_ALIGN OPENFILENAME ofn{};

	auto const& lwbType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& hueCoarse = params[COLOR_CORRECT_HUE_COARSE_LEVEL]->u.ad.value;
	auto const& hueFine   = params[COLOR_HUE_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& satCoarse = params[COLOR_SATURATION_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& satFine   = params[COLOR_SATURATION_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& lwbCoarse = params[COLOR_LWIP_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& lwbFine   = params[COLOR_LWIP_FINE_LEVEL_SLIDER]->u.fs_d.value;

	strParams.ver        = gVersion;
	strParams.subVer     = gSubVersion;
	strParams.sizeOf     = strHslRecorsSizeof;
	strncpy_s(strParams.name, strName, 32);
	strParams.name[31] = static_cast<char>(0);
	strParams.domain     = lwbType;
	strParams.hue_coarse = hueCoarse;
	strParams.hue_fine   = hueFine;
	strParams.sat_coarse = satCoarse;
	strParams.sat_file   = satFine;
	strParams.l_coarse   = lwbCoarse;
	strParams.l_fine     = lwbFine;

	/* get current time */
	auto const& now = std::chrono::system_clock::now();
	auto const& now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();

	/* get user home directory */
	std::string const& userHome{ desktop_directory ()};

	/* generate file name */
	std::ostringstream ossMessage;
	ossMessage << userHome << "\\ColorHSL_" << now_ms << ".lab2";
	std::string const& fileName{ ossMessage.str() };

	ofn.lStructSize = sizeof(ofn); // SEE NOTE BELOW
	ofn.hwndOwner = nullptr;// GetDesktopWindow();
	ofn.lpstrInitialDir = userHome.c_str();
	ofn.lpstrFilter = "Lab2 file (*.lab2)\0All Files (*.*)\0*.*\0";
	ofn.lpstrFile = const_cast<LPSTR>(fileName.c_str());
	ofn.nMaxFile = fileName.size();
	ofn.Flags = OFN_EXPLORER | OFN_SHOWHELP | OFN_OVERWRITEPROMPT;
	ofn.lpstrDefExt = "lab2";

	return false;
}


bool LoadCustomSetting
(
	PF_ParamDef* params[]
) noexcept
{
	CACHE_ALIGN strHslRecord strParams{};

	return false;
}


bool SaveCustomSetting
(
	PF_ParamDef* params[]
) noexcept
{
	bool bSave = false;

	if (1u == gVersion && 0u == gSubVersion)
	{
		bSave = SaveCustomSetting_1_0 (params);
	}

	return bSave;
}
