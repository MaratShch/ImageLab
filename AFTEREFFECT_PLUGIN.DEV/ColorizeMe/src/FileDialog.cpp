#include <string>
#include <windows.h>
#include "ColorizeMe.hpp"

const std::string GetLutFileName (void)
{
	std::string lutName{};
	char filename[MAX_PATH]{};
	OPENFILENAME ofn{};

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "LUT Files\0*.cube\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = "Select a LUT File";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	if (GetOpenFileNameA(&ofn))
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
