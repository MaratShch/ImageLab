#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include "ImageLabAverageFilter.h"

static float* fLogTable = nullptr;

inline void init_log10_table (float* pTable, const int& table_size)
{
	__VECTOR_ALIGNED__
	for (int i = 0; i < table_size; i++)
	{
		const float& ii = static_cast<const float>(i + 1);
		pTable[i] = fast_log10f(ii);
	}
	return;
}


inline float* alocate_log10_table (const int& table_size)
{
	const size_t bytesSyze = table_size * sizeof(float);
	const size_t alignment = static_cast<size_t>(size_mem_align);
	void* ptr = _aligned_malloc (bytesSyze, alignment);
	if (nullptr != ptr)
	{
		fLogTable = reinterpret_cast<float*>(ptr);
	}
	return fLogTable;
}


inline void free_log10_table (float* fLogTbl)
{
	if (nullptr != fLogTbl)
	{
		_aligned_free(fLogTbl);
		fLogTbl = nullptr;
	}

	return;
}


const float* get_log10_table_ptr (void)
{
	const float* ptr = fLogTable;
	return ptr;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			fLogTable = alocate_log10_table (alg10TableSize);
			if (nullptr != fLogTable)
			{
				init_log10_table (fLogTable, alg10TableSize);
			}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			free_log10_table (fLogTable);
			fLogTable = nullptr;
		break;
	
		default:
		break;
	}

	return TRUE;
}

