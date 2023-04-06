#pragma once

#include <windows.h>

typedef void*  (WINAPI *OpenMemInterface) (void);
typedef void   (WINAPI *CloseMemInterface)(void* p);
typedef int32_t(WINAPI *AllocMemBlock)    (void* pMemHandle, int32_t size, int32_t align, void** pMem);
typedef void   (WINAPI *FreeMemBlock)     (int32_t id);

typedef struct MemoryManagerInterface
{
	OpenMemInterface  MemoryInterfaceOpen;
	CloseMemInterface MemoryInterfaceClose;
	AllocMemBlock     MemoryInterfaceAllocBlock;
	FreeMemBlock      MemoryInterfacReleaseBlock;
	DWORD             _dbgLastError;
};

