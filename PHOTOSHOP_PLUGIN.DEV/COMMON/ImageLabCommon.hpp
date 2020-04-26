#ifndef __IMAGE_LAB_COMMON_INCLUDE_FILE__
#define __IMAGE_LAB_COMMON_INCLUDE_FILE__

#include "hw_platform.hpp"
#include "compile_time_utils.hpp"
#include "fast_computation.hpp"

/* include common ADOBE for PS files */
#include "PIDefines.h"
#include "PITypes.h"
#include "PIAbout.h"
#include "PIFilter.h"
#include "PIUtilities.h"


/* define PlugIn Entry Point */
#ifdef __cplusplus
extern "C" {
#endif

	DLLExport void PluginMain (const int16 selector, FilterRecordPtr filterParamBlock, intptr_t* pData, int16* pResult);

#ifdef __cplusplus
}
#endif


#endif // __IMAGE_LAB_COMMON_INCLUDE_FILE__