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


inline void AEFX_INIT_PARAM_STRUCTURE(PF_ParamDef& strDef, const PF_ParamFlags& paramFlag, const PF_ParamUIFlags& uiFlag) noexcept
{
	AEFX_CLR_STRUCT_EX(strDef);
	strDef.flags = paramFlag;
	strDef.ui_flags = uiFlag;
	return;
}


template <typename T>
inline void Image_SimpleCopy
(
	const T* __restrict srcBuffer,
	      T* __restrict dstBuffer,
	const int32_t&      height,
	const int32_t&      width,
	const int32_t&      src_line_pitch,
	const int32_t&      dst_line_pitch
) noexcept
{
	for (int32_t j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = srcBuffer + j * src_line_pitch;
		      T* __restrict pDstLine = dstBuffer + j * dst_line_pitch;
		__VECTORIZATION__
		for (int32_t i = 0; i < width; i++) { pDstLine[i] = pSrcLine[i]; }
	}
	return;
}



inline void
MakeParamCopy (PF_ParamDef* __restrict actual[], PF_ParamDef copy[], const int32_t& size) noexcept
{
	if (nullptr != actual && nullptr != copy && 0 < size)
	{
		for (int32_t idx = 0; idx < size; idx++)
			copy[idx] = *actual[idx];
	}
	return;
}

inline bool
IsDisabledUI (const PF_ParamUIFlags& uiFlag) noexcept
{
	return ((uiFlag & PF_PUI_DISABLED) ? true : false);
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