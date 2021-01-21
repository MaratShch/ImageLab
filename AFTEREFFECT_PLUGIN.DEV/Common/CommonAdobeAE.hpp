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


/* Entry point prototype for all AE PLUGINS */
#ifdef __cplusplus
extern "C" {
#endif

	DllExport
		PF_Err EntryPointFunc(
			PF_Cmd			cmd,
			PF_InData		*in_data,
			PF_OutData		*out_data,
			PF_ParamDef		*params[],
			PF_LayerDef		*output,
			void			*extra);

	DllExport
		PF_Err xGPUFilterEntry
		(
			PF_Cmd			cmd,
			PF_InData		*in_data,
			PF_OutData		*out_data,
			PF_ParamDef		*params[],
			PF_LayerDef		*output,
			void			*extra);

#ifdef __cplusplus
}
#endif

#endif /* __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__ */