#include "ImageLabBilateral.h"

extern float* __restrict pBuffer1;
extern float* __restrict pBuffer2;

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		{
			// allocate memory buffers for temporary procssing
			pBuffer1 = reinterpret_cast<float* __restrict>(allocCIELabBuffer(CIELabBufferSize));
			pBuffer2 = reinterpret_cast<float* __restrict>(allocCIELabBuffer(CIELabBufferSize));
#ifdef _DEBUG
			if (nullptr != pBuffer1 && nullptr != pBuffer2)
			{
				ZeroMemory(pBuffer1, CIELabBufferSize);
				ZeroMemory(pBuffer2, CIELabBufferSize);
			}
#endif
			CreateColorConvertTable();
			gaussian_weights();
		}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		{
			DeleteColorConvertTable();
			freeCIELabBuffer(pBuffer1);
			freeCIELabBuffer(pBuffer2);
			pBuffer1 = pBuffer2 = nullptr;;
		}
		break;

		default:
		break;
		}

	return TRUE;
}

