#include "ColorTemperature.hpp"
#include "ColorTemperatureGUI.hpp"
#include "AEFX_SuiteHelper.h"


PF_Err DrawEvent
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*event_extra
)
{
	DRAWBOT_DrawRef drawing_ref = nullptr;
	PF_Err err = PF_Err_NONE;

	/* get the drawing reference */
	if (PF_Err_NONE == (err = AEFX_SuiteScoper<PF_EffectCustomUISuite1>(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, out_data)->
		                PF_GetDrawingReference (event_extra->contextH, &drawing_ref)))
	{
		DRAWBOT_SupplierRef	supplier_ref = nullptr;
		DRAWBOT_SurfaceRef	surface_ref  = nullptr;

		/* acquire DrawBot Suite	*/
		auto const& drawbotSuite{ AEFX_SuiteScoper<DRAWBOT_DrawbotSuite1>(in_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, out_data) };

		/* acquire DrawBot Supplier and Surface; it shouldn't be released like pen or brush */
		auto const err1 = drawbotSuite->GetSupplier(drawing_ref, &supplier_ref);
		auto const err2 = drawbotSuite->GetSurface (drawing_ref, &surface_ref );
		if (kSPNoError == err1 && kSPNoError == err2 && nullptr != supplier_ref && nullptr != surface_ref)
		{

		}/* if (kASNoError == err1 && kASNoError == err2 && nullptr != supplier_ref && nullptr != surface_ref) */

	} /* if (PF_Err_NONE == (err = AEFX_SuiteScoper<PF_EffectCustomUISuite1> ...  */

	if (PF_Err_NONE == err)
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;

	return err;
}