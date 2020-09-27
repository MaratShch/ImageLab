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

#define PS_PLUGIN_VENDOR_NAME		"ImageLab"

#define	PS_PLUGIN_ENTRY_POINT		"PluginMain"
#define	PS_PLUGIN_ENTRY_POINT_MAC	PS_PLUGIN_ENTRY_POINT
#define	PS_PLUGIN_ENTRY_POINT_WIN32	PS_PLUGIN_ENTRY_POINT
#define	PS_PLUGIN_ENTRY_POINT_WIN64	PS_PLUGIN_ENTRY_POINT

#define PS_PLUGIN_COPYRIGHT_YEAR	"2020"

#define PS_PLUGIN_MAJOR_VERSION		1


/* define PlugIn Entry Point */
#ifdef __cplusplus
extern "C" {
#endif

	DLLExport void PluginMain (const int16 selector, FilterRecordPtr filterParamBlock, intptr_t* pData, int16* pResult);

#ifdef __cplusplus
}
#endif


#endif // __IMAGE_LAB_COMMON_INCLUDE_FILE__