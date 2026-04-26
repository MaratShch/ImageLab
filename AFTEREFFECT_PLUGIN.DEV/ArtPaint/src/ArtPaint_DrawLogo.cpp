#include <string>
#include "ArtPaintResource.hpp"
#include "ArtPaint_DrawLogo.hpp"
#include "CommonAdobeAE.hpp"


static HMODULE hLib = NULL;
static CACHE_ALIGN Logo logoBmp{};


const Logo& getBitmap(void)
{
    return logoBmp;
}


bool LoadResourceDll (PF_InData* in_data)
{
    A_char pluginFullPath[AEFX_MAX_PATH]{};
    PF_Err extErr = PF_GET_PLATFORM_DATA(PF_PlatData_EXE_FILE_PATH_DEPRECATED, &pluginFullPath);
    bool err = false;

    if (PF_Err_NONE == extErr && 0 != pluginFullPath[0])
    {
        const std::string dllName{ "\\ImageLabResource.dll" };
        const std::string aexPath{ pluginFullPath };
        const std::string::size_type pos = aexPath.rfind("\\", aexPath.length());
        const std::string dllPath = aexPath.substr(0, pos) + dllName;

        // Load Memory Management DLL
        hLib = ::LoadLibraryEx(dllPath.c_str(), NULL, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (NULL != hLib)
            err = true;
    }

    return err;
}

void FreeResourceDll(void)
{
    if (NULL != hLib)
    {
        ::FreeLibrary(hLib);
        hLib = NULL;
    }
    return;
}


bool LoadLogo (void)
{
    bool bRet = false;
    if (NULL != hLib)
    {
        // Load bitmap from resource
        HBITMAP hBmp = static_cast<HBITMAP>(LoadImage(hLib, MAKEINTRESOURCE(IDB_BITMAP_LOGO_ART_PAINT), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION));
        if (nullptr != hBmp)
        {
            BITMAP bmp{};
            GetObject(hBmp, sizeof(BITMAP), &bmp);

            if (bmp.bmBits)
            {
                std::memcpy(logoBmp.data(), bmp.bmBits, logoBmp.size());
            } // if (bmp.bmBits)

            DeleteObject(hBmp);
            hBmp = nullptr;
            bRet = true;
        }
    }
    return bRet;
}

