#include <string>
#include <windows.h>
#include "ColorizeMe.hpp"

static constexpr char dialogTitle[] = "Select a LUT File";

const std::string GetLutFileName (void)
{
	std::string lutName;
	char filename[MAX_PATH]{};
	OPENFILENAME ofn{};

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "CUBE Files\0*.cube\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
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
