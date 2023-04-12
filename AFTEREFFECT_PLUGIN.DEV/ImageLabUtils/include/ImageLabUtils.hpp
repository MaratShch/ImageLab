#pragma once

#include <stdint.h>
#include "LibExport.hpp"

DLL_API_EXPORT void*   CreateMemoryHandler (void);
DLL_API_EXPORT void    ReleaseMemoryHandler(void* p);
DLL_API_EXPORT void    ReleaseMemoryBlock  (void* pMemHandle, int32_t id);
DLL_API_EXPORT int32_t AllocMemoryBlock    (void* pMemHandle, int32_t size, int32_t align, void** pMem);
DLL_API_EXPORT int64_t GetMemoryStatistics (void* pMemHandle);
