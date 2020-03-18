#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include "ImageLabAverageFilter.h"

static float* __restrict fLogTable;

inline void init_1og10_table(float* pTable, int table_size)
{
	__VECTOR_ALIGNED__
	for (int i = 0; i < table_size; i++)
	{
		const float& ii = static_cast<const float>(i + 1);
		pTable[i] = fast_log10f(ii);
	}
	return;
}


inline float* alocate_log10_table(const int& table_size)
{
	if (0 >= table_size)
		return nullptr;

	return reinterpret_cast<float*>(_aligned_malloc(static_cast<size_t>(table_size), static_cast<size_t>(size_mem_align)));
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


const float* get_log10_table_ptr(void)
{
	return fLogTable;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			fLogTable = alocate_log10_table (alg10TableSize);
			if (nullptr != fLogTable)
			{
				init_1og10_table (fLogTable, alg10TableSize);
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

