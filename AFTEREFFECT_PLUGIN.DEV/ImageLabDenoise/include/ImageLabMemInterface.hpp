#ifndef __IMAGE_LAB_RETRO_VISION_EXT_MEM_INTERFACE__
#define __IMAGE_LAB_RETRO_VISION_EXT_MEM_INTERFACE__

#include <windows.h>
#include "AE_Effect.h"

bool LoadMemoryInterfaceProvider (PF_InData* in_data);
int32_t GetMemoryBlock (int32_t size, int32_t align, void** pMem) noexcept;
void FreeMemoryBlock (int32_t id) noexcept;
int32_t memGetLastError (void) noexcept;
void UnloadMemoryInterfaceProvider(void);


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
} MemoryManagerInterface;

#endif // __IMAGE_LAB_RETRO_VISION_EXT_MEM_INTERFACE__