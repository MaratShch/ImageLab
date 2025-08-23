#include "CompileTimeUtils.hpp"
#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionGui.hpp"
#include "DrawbotSuite.h"
#include "AEFX_SuiteHelper.h"
#include <cmath>
#include <atomic>

#ifdef _DEBUG
#include <string>
#include <sstream>

static DRAWBOT_Boolean supportsBGRA = false; /* true */
static DRAWBOT_Boolean prefersBGRA = false;
static DRAWBOT_Boolean supportsARGB = false; /* false */
static DRAWBOT_Boolean prefersARGB = false;
static std::atomic<bool>preferredFormatDecided{ false };
#endif



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
    DRAWBOT_SupplierRef	supplier_ref = nullptr;
    DRAWBOT_SurfaceRef	surface_ref = nullptr;

    PF_Err err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    // acquire DrawBot Suites
    auto const drawbotSuite  = AEFX_SuiteScoper<DRAWBOT_DrawbotSuite1> (in_data, kDRAWBOT_DrawSuite,     kDRAWBOT_DrawSuite_VersionCurrent,     out_data);
    auto const supplierSuite = AEFX_SuiteScoper<DRAWBOT_SupplierSuite1>(in_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, out_data);

    auto const effectCustomUISuiteP = AEFX_SuiteScoper<PF_EffectCustomUISuite1>(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, out_data);

    // Get the drawing reference by passing context to this new api
    const PF_Err ErrDrawRef = effectCustomUISuiteP->PF_GetDrawingReference(event_extra->contextH, &drawing_ref);

    // Get the Drawbot supplier from drawing reference; it shouldn't be released like pen or brush (see below)
    const PF_Err ErrSupplRef = drawbotSuite->GetSupplier(drawing_ref, &supplier_ref);

    DRAWBOT_ImageRef drawbotImage{};

    err = supplierSuite->NewImageFromBuffer
    (
        supplier_ref,
        48,48, 48*4,
        kDRAWBOT_PixelLayout_32ARGB_Straight,
        getBitmap().data(),
        &drawbotImage
    );

    auto const surfaceSuite = AEFX_SuiteScoper<DRAWBOT_SurfaceSuite1> (in_data, kDRAWBOT_SurfaceSuite,  kDRAWBOT_SurfaceSuite_VersionCurrent,  out_data);
    // Get DarwBot Surface
    const SPErr ErrSurface = drawbotSuite->GetSurface(drawing_ref, &surface_ref);
    DRAWBOT_PointF32 drawbotPoint{ static_cast<float>(event_extra->effect_win.current_frame.left) + 0.5f, static_cast<float>(event_extra->effect_win.current_frame.top) + 0.5f};
    float alpha = 1.f;

    surfaceSuite->DrawImage(surface_ref, drawbotImage, &drawbotPoint, alpha);

    return err;
}