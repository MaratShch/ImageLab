#pragma once

#include <windows.h>

typedef void*  (WINAPI *OpenMemInterface) (void);
typedef void   (WINAPI *CloseMemInterface)(void* p);
typedef int32_t(WINAPI *AllocMemBlock)    (void* pMemHandle, int32_t size, int32_t align, void** pMem);
typedef void   (WINAPI *FreeMemBlock)     (void* pMemHandle, int32_t id);

typedef struct MemoryManagerInterface
{
	OpenMemInterface  MemoryInterfaceOpen;
	CloseMemInterface MemoryInterfaceClose;
	AllocMemBlock     MemoryInterfaceAllocBlock;
	FreeMemBlock      MemoryInterfaceReleaseBlock;
	DWORD             _dbgLastError;
};

bool LoadMemoryInterfaceProvider(int32_t appId, int32_t major, int32_t minor = 0) noexcept;
int32_t GetMemoryBlock(int32_t size, int32_t align, void** pMem) noexcept;
void FreeMemoryBlock(int32_t id) noexcept;