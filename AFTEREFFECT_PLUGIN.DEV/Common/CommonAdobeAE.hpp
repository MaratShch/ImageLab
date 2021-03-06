#ifndef __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__
#define __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__

#include "AEConfig.h"
#include "entry.h"
#ifdef AE_OS_WIN
#include "string.h"
#endif
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "AE_EffectCBSuites.h"
#include "AE_GeneralPlug.h"
#include "AEFX_SuiteHandlerTemplate.h"

#include "Common.hpp"
#include "Param_Utils.h"
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"

#ifdef _DEBUG
#define PF_DISPOSE_HANDLE_EX(PF_HANDLE)                     \
    memset(*(PF_HANDLE), 0, PF_GET_HANDLE_SIZE(PF_HANDLE)); \
    PF_DISPOSE_HANDLE(PF_HANDLE);							
#else
#define PF_DISPOSE_HANDLE_EX(PF_HANDLE)                     \
    memset(*(PF_HANDLE), 0, PF_GET_HANDLE_SIZE(PF_HANDLE)); \
    PF_DISPOSE_HANDLE(PF_HANDLE);							\
    (PF_HANDLE) = nullptr;
#endif

#ifndef GET_OBJ_FROM_HNDL
 #define GET_OBJ_FROM_HNDL(h) (*(h))
#endif


inline void
MakeParamCopy (PF_ParamDef* actual[], PF_ParamDef copy[], const uint32_t& size) noexcept
{
	for (uint32_t idx = 0u; idx < size; idx++)
		copy[idx] = *actual[idx];
	return;
}


#ifdef __cplusplus
 #define PLUGIN_ENTRY_POINT_CALL	extern "C" DllExport
#else
 #define PLUGIN_ENTRY_POINT_CALL DllExport
#endif

/* Entry point prototype for all AE PLUGINS */
	PLUGIN_ENTRY_POINT_CALL
		PF_Err EffectMain (
			PF_Cmd			cmd,
			PF_InData		*in_data,
			PF_OutData		*out_data,
			PF_ParamDef		*params[],
			PF_LayerDef		*output,
			void			*extra);

#endif /* __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__ */