﻿#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <Windows.h>
#include "ImageLabBilateral.h"


BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
    {
		case DLL_PROCESS_ATTACH:
			gaussian_weights();
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		break;

		default:
		break;
    }

    return TRUE;
}

