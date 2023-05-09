#include <string>
#include <windows.h>
#include "ColorizeMe.hpp"

static constexpr char dialogTitle[] = "Select a LUT File";
constexpr size_t filePathSize = MAX_PATH;

const std::string GetLutFileName (void)
{
	std::string lutName;
	TCHAR filename[filePathSize]{};
	OPENFILENAME ofn{};
	constexpr size_t ofnStrSize = sizeof(ofn);

	ofn.lStructSize = ofnStrSize;
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "CUBE Files\0*.cube\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = filePathSize;
	ofn.lpstrTitle = dialogTitle;
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	if (TRUE == GetOpenFileNameA (&ofn))
		lutName = filename;
	else
		lutName.clear();

	return lutName;
}

const std::string GetLutFileName(const std::string& fileMask)
{
	std::string lutName{};
	return lutName;
}
