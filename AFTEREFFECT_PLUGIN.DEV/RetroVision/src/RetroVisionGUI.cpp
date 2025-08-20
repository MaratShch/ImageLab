#include "CompileTimeUtils.hpp"
#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "AEFX_SuiteHelper.h"
#include <cmath>

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
    DRAWBOT_PathRef		path_ref = nullptr;
    DRAWBOT_BrushRef	brush_ref = nullptr;

//    if (false == isRedraw())
//        return PF_Err_NONE;

    PF_Err err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    // Let's notify that flag for redraw received successfully.
//    ProcRedrawComplete();

    // acquire DrawBot Suites
    auto const drawbotSuite { AEFX_SuiteScoper<DRAWBOT_DrawbotSuite1> (in_data, kDRAWBOT_DrawSuite,     kDRAWBOT_DrawSuite_VersionCurrent,     out_data) };
    auto const supplierSuite{ AEFX_SuiteScoper<DRAWBOT_SupplierSuite1>(in_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, out_data) };
    auto const surfaceSuite { AEFX_SuiteScoper<DRAWBOT_SurfaceSuite1> (in_data, kDRAWBOT_SurfaceSuite,  kDRAWBOT_SurfaceSuite_VersionCurrent,  out_data) };
    auto const pathSuite    { AEFX_SuiteScoper<DRAWBOT_PathSuite1>    (in_data, kDRAWBOT_PathSuite,     kDRAWBOT_PathSuite_VersionCurrent,     out_data) };

    // get the drawing reference
    auto const effectCustomUISuiteP{ AEFX_SuiteScoper<PF_EffectCustomUISuite1>(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, out_data) };

    // Get the drawing reference by passing context to this new api
    const PF_Err ErrDrawRef = effectCustomUISuiteP->PF_GetDrawingReference(event_extra->contextH, &drawing_ref);

    // Get the Drawbot supplier from drawing reference; it shouldn't be released like pen or brush (see below)
    const PF_Err ErrSupplRef = drawbotSuite->GetSupplier(drawing_ref, &supplier_ref);

    // Get the Drawbot surface from drawing reference; it shouldn't be released like pen or brush (see below)
    const PF_Err ErrSurfRef = drawbotSuite->GetSurface(drawing_ref, &surface_ref);

#ifdef _DEBUG
    if (false == preferredFormatDecided)
    {
        preferredFormatDecided = true;
        supplierSuite->SupportsPixelLayoutBGRA(supplier_ref, &supportsBGRA);
        supplierSuite->PrefersPixelLayoutBGRA(supplier_ref, &prefersBGRA);
        supplierSuite->SupportsPixelLayoutARGB(supplier_ref, &supportsARGB);
        supplierSuite->PrefersPixelLayoutARGB(supplier_ref, &prefersARGB);

        if (PremierId == in_data->appl_id)
        {
            // In Premiere Pro, this message will appear in the Events panel
            std::ostringstream dbgStr;
            dbgStr << "supportsBGRA = " << (true == supportsBGRA ? "1" : "0") << " supportsARGB = " << (true == supportsARGB ? "1" : "0");
            std::string finalStr = dbgStr.str();
            PF_STRCPY(out_data->return_msg, finalStr.data());
        }
    } // if (false == preferredFormatDecided)
#endif

    if (PF_Err_NONE == ErrDrawRef && kSPNoError == ErrSupplRef && kSPNoError == ErrSurfRef)
    {
        DRAWBOT_ColorRGBA drawbot_color{};

        // Premiere Pro/Elements does not support a standard parameter type
        // with custom UI (bug #1235407), so we can't use the color values.
        // Use an static grey value instead.
        if (PremierId != in_data->appl_id)
        {
            constexpr float PfMaxChan8 = { static_cast<float>(u8_value_white) };
            constexpr auto GuiControl = UnderlyingType(RetroVision::eRETRO_VISION_GUI);
            drawbot_color.red = static_cast<float>  (params[GuiControl]->u.cd.value.red)   / PfMaxChan8;
            drawbot_color.green = static_cast<float>(params[GuiControl]->u.cd.value.green) / PfMaxChan8;
            drawbot_color.blue = static_cast<float> (params[GuiControl]->u.cd.value.blue)   / PfMaxChan8;
        }
        else
        {
            static float gray{ 0.f };
            drawbot_color.red   = std::fmod(gray, 1.f);
            drawbot_color.green = std::fmod(gray, 1.f);
            drawbot_color.blue  = std::fmod(gray, 1.f);
            gray += 0.01f;
        }
        drawbot_color.alpha = 1.0f;

        if (PF_EA_CONTROL == event_extra->effect_win.area)
        {
            // Create a new path. It should be matched with release routine.
            // You can also use C++ style DRAWBOT_PathP that releases automatically at the end of scope.
            if (kSPNoError == supplierSuite->NewPath(supplier_ref, &path_ref))
            {
                // Create a new brush taking color as input; it should be matched with release routine.
                // You can also use C++ style DRAWBOT_BrushP which doesn't require release routine.
                if (kSPNoError == supplierSuite->NewBrush(supplier_ref, &drawbot_color, &brush_ref))
                {
                    DRAWBOT_RectF32	rectR{};

                    rectR.left   = static_cast<float>(event_extra->effect_win.current_frame.left) + 0.5f;
                    rectR.top    = static_cast<float>(event_extra->effect_win.current_frame.top ) + 0.5f;
                    rectR.width  = static_cast<float>(event_extra->effect_win.current_frame.right  - event_extra->effect_win.current_frame.left);
                    rectR.height = static_cast<float>(event_extra->effect_win.current_frame.bottom - event_extra->effect_win.current_frame.top);

                    // Add the rectangle to path
                    if (kSPNoError == pathSuite->AddRect(path_ref, &rectR))
                    {
                        // Fill the path with the brush created
                        if (kSPNoError == surfaceSuite->FillPath(surface_ref, brush_ref, path_ref, kDRAWBOT_FillType_Default))
                        {

                            // event successfully processed and released
                            event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;
                            err = PF_Err_NONE;
                        } // if (kSPNoError == surfaceSuite->FillPath(surface_ref, brush_ref, path_ref, kDRAWBOT_FillType_Default))

                    } // if (kSPNoError == pathSuite->AddRect(path_ref, &rectR))

                } // if (kSPNoError == supplierSuite->NewBrush(supplier_ref, &drawbot_color, &brush_ref))

            } // if (kSPNoError == supplierSuite->NewPath(supplier_ref, &path_ref))

        } // if (PF_EA_CONTROL == event_extra->effect_win.area)
    } // if (PF_Err_NONE == ErrDrawRef && kSPNoError == ErrSupplRef && kSPNoError == ErrSurfRef)

    return err;
}