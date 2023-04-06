#include <windows.h>
#include <Libloaderapi.h>
#include "Common.hpp"
#include "ImageEqualization.hpp"
#include <string>

static HINSTANCE hLib = NULL;
static MemoryManagerInterface memInterface{};
static void* MemoryInterfaceHndl = nullptr;


static std::string GetStringRegKey (const std::string& key, const std::string& subKey)
{
	CACHE_ALIGN TCHAR szBuffer[256]{};
	std::string keyValue;
	LONG err = ERROR_SUCCESS;
	HKEY hKey = 0;

	DWORD dwBufferSize = sizeof(szBuffer);
	if (ERROR_SUCCESS == (err = RegOpenKeyEx(HKEY_LOCAL_MACHINE, key.c_str(), 0, KEY_QUERY_VALUE, &hKey)))
	{
		if (ERROR_SUCCESS == (err = RegQueryValueEx(hKey, subKey.c_str(), 0, NULL, (LPBYTE)szBuffer, &dwBufferSize)))
			keyValue = szBuffer;
		RegCloseKey(hKey);
		hKey = 0;
	}
	else
	{
		keyValue = ".\\";
	}

	return keyValue;
}


static void InitializeMemoryUtilsInterface (const HINSTANCE h)
{
	if (NULL != h)
	{
		memInterface.MemoryInterfaceOpen  = reinterpret_cast<OpenMemInterface>    (GetProcAddress(h, __TEXT("CreateMemoryHandler")));
		memInterface.MemoryInterfaceClose = reinterpret_cast<CloseMemInterface>   (GetProcAddress(h, __TEXT("ReleaseMemoryHandler")));
		memInterface.MemoryInterfaceAllocBlock   = reinterpret_cast<AllocMemBlock>(GetProcAddress(h, __TEXT("AllocMemoryBlock")));
		memInterface.MemoryInterfaceReleaseBlock = reinterpret_cast<FreeMemBlock> (GetProcAddress(h, __TEXT("ReleaseMemoryBlock")));

		if (NULL != memInterface.MemoryInterfaceOpen)
		{
			/* open memory interface handler */
			MemoryInterfaceHndl = memInterface.MemoryInterfaceOpen();
		}
	}
	memInterface._dbgLastError = GetLastError();

	return;
}

bool LoadMemoryInterfaceProvider (int32_t appId, int32_t major, int32_t minor)
{
	const std::string dllName = __TEXT("\\ImageLab2\\ImageLabUtils.dll");
	std::string keyValue;

	if (PremierId == appId)
	{
		/* we started from Premier. Let's search the Premier Plugin Common folder */
		std::string key = __TEXT("SOFTWARE\\Adobe\\Premiere Pro\\CurrentVersion");
		std::string subKey = __TEXT("Plug-InsDir");
		keyValue = GetStringRegKey(key, subKey);
	}
	else
	{
		/* we started from AfterEffect. Let's search the After Effect Plugin common folder */
//		subKey = "SOFTWARE\\Adobe\\After Effects\\" + std::to_string(major) + "." + std::to_string(minor) + "\\CommonPluginInstallPath";
	}

		std::string fullDllPath = keyValue + dllName;
		hLib = ::LoadLibraryEx(__TEXT(fullDllPath.c_str()), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
		memInterface._dbgLastError = GetLastError();
		InitializeMemoryUtilsInterface (hLib);

	return true;
}

int32_t GetMemoryBlock (int32_t size, int32_t align, void** pMem)
{
	if (NULL != hLib && NULL != memInterface.MemoryInterfaceAllocBlock && NULL != MemoryInterfaceHndl && nullptr != pMem)
		return memInterface.MemoryInterfaceAllocBlock (MemoryInterfaceHndl, size, align, pMem);
	return -1;
}

void FreeMemoryBlock (int32_t id)
{
	if (NULL != hLib && NULL != memInterface.MemoryInterfaceReleaseBlock && NULL != MemoryInterfaceHndl && id >= 0)
		memInterface.MemoryInterfaceReleaseBlock(MemoryInterfaceHndl, id);
	id = -1;
}

// Computer\HKEY_LOCAL_MACHINE\SOFTWARE\Adobe\After Effects\14.0
// CommonPluginInstallPath = C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore

BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	constexpr size_t memInterfaceSize = sizeof(memInterface);

	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_DETACH:
			FreeLibrary(hLib);
			memset(&memInterface, 0, memInterfaceSize);
			hLib = NULL;
		break;

		case DLL_PROCESS_ATTACH:
			hLib = NULL;
			MemoryInterfaceHndl = nullptr;
			memset(&memInterface, 0, memInterfaceSize);
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		default:
		break;
	}

	return TRUE;
}

