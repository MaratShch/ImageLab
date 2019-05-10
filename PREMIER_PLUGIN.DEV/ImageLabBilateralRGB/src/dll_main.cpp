#include "ImageLabBilateral.h"


inline void* allocCIELabBuffer(const size_t& size)
{
	void* pMem = _aligned_malloc(size, CIELabBufferAlign);
	if (nullptr != pMem)
	{
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
	}
	return pMem;
}

inline void freeCIELabBuffer(void* pMem)
{
	if (nullptr != pMem)
	{
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
		_aligned_free(pMem);
		pMem = nullptr;
	}
}

inline int numCpuCores(void)
{
	SYSTEM_INFO sysinfo = { 0 };
	GetSystemInfo(&sysinfo);
	return static_cast<int>(sysinfo.dwNumberOfProcessors);
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{

	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		{
			CreateColorConvertTable();
		}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		{
			DeleteColorConevrtTable();
		}
		break;

		default:
		break;
		}

	return TRUE;
}

