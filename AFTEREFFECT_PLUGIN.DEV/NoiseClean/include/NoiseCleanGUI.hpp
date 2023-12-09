#ifndef __NOISE_CLEAN_GUI_ELEMENTS_MANAGEMENT__
#define __NOISE_CLEAN_GUI_ELEMENTS_MANAGEMENT__

#include "PrSDKAESupport.h"

void SwitchToNoAlgo
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept;

void SwitchToBilateral
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept;

void SwitchToAnysotropic
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept;


#endif  /* __NOISE_CLEAN_GUI_ELEMENTS_MANAGEMENT__ */