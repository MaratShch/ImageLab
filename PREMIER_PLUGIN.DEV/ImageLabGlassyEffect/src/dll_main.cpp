#include "ImageLabGlassyEffect.h"

static float* pRandomValBuffer = nullptr;

static unsigned int utils_get_random_value (void)
{
	// used xorshift algorithm allow for this random generator pass Diehard Tests 
	static unsigned int x = 123456789u;
	static unsigned int y = 362436069u;
	static unsigned int z = 521288629u;
	static unsigned int w = 88675123u;
	static unsigned int t;

	t = x ^ (x << 11);
	x = y; y = z; z = w;
	return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}


static void generateRandowValues(float* pBuffer, const size_t& bufSize)
{
	const int samples = static_cast<int>(bufSize);

	if (nullptr != pBuffer && 0 != samples)
	{
		for (int i = 0; i < samples; i++)
		{
			pBuffer[i] = static_cast<float>(utils_get_random_value()) / 4294967295.0f;
		}
	}

	return;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		{
			pRandomValBuffer = reinterpret_cast<float*>(_aligned_malloc(randomBufSize * sizeof(float), CACHE_LINE));
			generateRandowValues(pRandomValBuffer, randomBufSize);
		}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		{
			if (nullptr != pRandomValBuffer)
			{
				// for DBG purpose
				ZeroMemory(pRandomValBuffer, randomBufSize * sizeof(float));
				_aligned_free(reinterpret_cast<void*>(pRandomValBuffer));
				pRandomValBuffer = nullptr;
			}
		}
		break;

		default:
		break;
		}

	return TRUE;
}

