#ifndef __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__
#define __IMAGE_LAB2_ADOBE_AE_COMMON_INCLUDES_FILES__

#include <cmath>

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
	const int32_t      height,
	const int32_t      width,
	const int32_t      src_line_pitch,
	const int32_t      dst_line_pitch
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
MakeParamCopy (PF_ParamDef* __restrict actual[], PF_ParamDef copy[], const int32_t size) noexcept
{
	if (nullptr != actual && nullptr != copy && 0 < size)
	{
		for (int32_t idx = 0; idx < size; idx++)
			copy[idx] = *actual[idx];
	}
	return;
}

inline constexpr bool
IsDisabledUI (const PF_ParamUIFlags uiFlag) noexcept
{
	return ((uiFlag & PF_PUI_DISABLED) ? true : false);
}

inline void DisableUI (PF_ParamUIFlags uiFlag) noexcept
{
	uiFlag |= PF_PUI_DISABLED;
	return;
}

inline void EnableUI (PF_ParamUIFlags uiFlag) noexcept
{
	uiFlag &= ~PF_PUI_DISABLED;
	return;
}


inline double image_lab_get_fps (const PF_InData* in_data) noexcept
{
    if (nullptr == in_data)
        return 0.0;

    // Prefer local_time_step: documented as "constant from one frame to the
    // next", where time_step can vary per frame under time remapping.
    const A_long step = (0 == in_data->local_time_step) ? in_data->time_step : in_data->local_time_step;

    if (0 == step || 0u == in_data->time_scale)   // both can legitimately be 0
        return 0.0;

    // Convert before taking the magnitude. std::abs on an integer needs
    // <cstdlib> for its guaranteed overloads, and std::abs(INT32_MIN) is UB --
    // measured, it returns INT32_MIN itself, i.e. still negative, which would
    // silently invert the sign of fps. Negative step is legal here: it means
    // the layer is time-reversed.
    const double stepMag = std::abs(static_cast<double>(step));

    double fps = static_cast<double>(in_data->time_scale) / stepMag;

    // Premiere Pro sends PF_Cmd_RENDER once per FIELD in native pixel formats,
    // so the ratio above yields the FIELD rate (60000/1001 = 59.94 for NTSC),
    // not the frame rate. Both fields of one frame must share the same dust and
    // scratches or the damage strobes at 60 Hz.
    if (PremierId == in_data->appl_id && PF_Field_FRAME != in_data->field)
        fps *= 0.5;

    return fps;
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