#include <windows.h>
#include <Libloaderapi.h>
#include <string>
#include "NoiseClean.hpp"
#include "ImageLabMemInterface.hpp"

static MemoryManagerInterface memInterface{};
static HMODULE hLib = NULL;
static void* MemoryInterfaceHndl = nullptr;


static std::string GetStringRegKey(const std::string& key, const std::string& subKey) noexcept
{
	CACHE_ALIGN TCHAR szBuffer[_MAX_PATH]{};
	std::string keyValue;
	LONG err = ERROR_SUCCESS;
	HKEY hKey = 0;

	DWORD dwBufferSize = sizeof(szBuffer);
	if (ERROR_SUCCESS == (err = RegOpenKeyEx(HKEY_LOCAL_MACHINE, key.c_str(), 0, KEY_QUERY_VALUE, &hKey)))
	{
		if (ERROR_SUCCESS == (err = RegQueryValueEx(hKey, subKey.c_str(), 0, NULL, reinterpret_cast<LPBYTE>(szBuffer), &dwBufferSize)))
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


static void InitializeMemoryUtilsInterface(const HINSTANCE h) noexcept
{
	if (NULL != h)
	{
		memInterface.MemoryInterfaceOpen = reinterpret_cast<OpenMemInterface>    (GetProcAddress(h, __TEXT("CreateMemoryHandler")));
		memInterface.MemoryInterfaceClose = reinterpret_cast<CloseMemInterface>  (GetProcAddress(h, __TEXT("ReleaseMemoryHandler")));
		memInterface.MemoryInterfaceAllocBlock = reinterpret_cast<AllocMemBlock> (GetProcAddress(h, __TEXT("AllocMemoryBlock")));
		memInterface.MemoryInterfaceReleaseBlock = reinterpret_cast<FreeMemBlock>(GetProcAddress(h, __TEXT("ReleaseMemoryBlock")));

		if (NULL != memInterface.MemoryInterfaceOpen)
		{
			/* open memory interface handler */
			MemoryInterfaceHndl = memInterface.MemoryInterfaceOpen();
		}
	}
	memInterface._dbgLastError = GetLastError();

	return;
}

inline std::string AppId2Version(const int32_t& major, const int32_t& minor) noexcept
{
	std::string ver = "14.0"; /* tmp workaround */
	return ver;
}


bool LoadMemoryInterfaceProvider(int32_t appId, int32_t major, int32_t minor) noexcept
{
	HMODULE h = 0;
	const std::string dllName{ "ImageLabUtils.dll" };
	const std::string dllCategory{ "\\ImageLab2\\" };
	std::string keyValue;

	if (PremierId == appId)
	{
		/* we started from Premier. Let's search the Premier Plugin Common folder */
		const std::string key{ "SOFTWARE\\Adobe\\Premiere Pro\\CurrentVersion" };
		const std::string subKey{ "Plug-InsDir" };
		keyValue = GetStringRegKey(key, subKey);
	}
	else
	{
		/* we started from AfterEffect. Let's search the After Effect Plugin common folder */
		const std::string key{ "SOFTWARE\\Adobe\\After Effects\\" + AppId2Version(major, minor) };
		const std::string subKey{ "CommonPluginInstallPath" };
		keyValue = GetStringRegKey(key, subKey);
	}

	const std::string fullDllPath = keyValue + dllCategory + dllName;
	hLib = ::LoadLibraryEx(fullDllPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
	memInterface._dbgLastError = GetLastError();

#ifdef _DEBUG
	const BOOL bDisableThreadCall =
#endif
	
	DisableThreadLibraryCalls(hLib);
	InitializeMemoryUtilsInterface(hLib);

	return true;
}

int32_t GetMemoryBlock(int32_t size, int32_t align, void** pMem) noexcept
{
	if (NULL != hLib && NULL != memInterface.MemoryInterfaceAllocBlock && NULL != MemoryInterfaceHndl && nullptr != pMem)
		return memInterface.MemoryInterfaceAllocBlock(MemoryInterfaceHndl, size, align, pMem);
	return -1;
}

void FreeMemoryBlock(int32_t id) noexcept
{
	if (NULL != hLib && NULL != memInterface.MemoryInterfaceReleaseBlock && NULL != MemoryInterfaceHndl && id >= 0)
		memInterface.MemoryInterfaceReleaseBlock(MemoryInterfaceHndl, id);
	id = -1;
	return;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	constexpr size_t memInterfaceSize = sizeof(memInterface);

	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_DETACH:
			if (NULL != hLib)
			{
				FreeLibrary(hLib);
				hLib = NULL;
				MemoryInterfaceHndl = nullptr;
				memset(&memInterface, 0, memInterfaceSize);
			}
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

