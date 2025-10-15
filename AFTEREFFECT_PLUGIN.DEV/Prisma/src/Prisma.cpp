#include "Prisma.hpp"
#include "PrismaVulkan.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"
#include "ImageLabVulkanLoader.hpp"

static void* vkAlgoHandler{ nullptr };

static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(
		out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		PrismaVideo_VersionMajor,
		PrismaVideo_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    PF_Err	err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    // Load Vulkan algorithms library
    if (false == LoadVulkanAlgoDll(in_data))
        return err;

    // Load memory interface for alloc temporary buffers
    if (false == LoadMemoryInterfaceProvider(in_data))
    {
        vkAlgoHandler = nullptr;
        UnloadVulkanAlgoDll ();
        return err;
    }

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_PIX_INDEPENDENT       |
		PF_OutFlag_SEND_UPDATE_PARAMS_UI |
		PF_OutFlag_USE_OUTPUT_EXTENT     |
		PF_OutFlag_DEEP_COLOR_AWARE      |
		PF_OutFlag_WIDE_TIME_INPUT;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

	out_data->my_version =
		PF_VERSION(
			PrismaVideo_VersionMajor,
			PrismaVideo_VersionMinor,
			PrismaVideo_VersionSub,
			PrismaVideo_VersionStage,
			PrismaVideo_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

    vkAlgoHandler = VulkanAllocNode (0, 0, 0);
    err = ((nullptr != vkAlgoHandler) ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);

	return err;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    // Free Vulkan Algorithm Handler 
    if (nullptr != vkAlgoHandler)
    {
        VulkanFreeNode(vkAlgoHandler);
        vkAlgoHandler = nullptr;
    }

    // Unload memory interface
    UnloadMemoryInterfaceProvider();

    // Unload Vuokan algorithms library 
    UnloadVulkanAlgoDll();

    return PF_Err_NONE;
}



static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{

	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}



static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*output
)
{
    PF_Err	err = PF_Err_NONE;
    return err;
}


static PF_Err SmartRender(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    return Prisma_SmartRender (in_data, out_data, extraP);
}


static PF_Err
PreRender(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_PreRenderExtra		*extraP
)
{
    return Prisma_PreRender(in_data, out_data, extraP);
}




PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain(
    PF_Cmd			cmd,
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    void			*extra)
{
    PF_Err		err{ PF_Err_NONE };

    try {
        switch (cmd)
        {
            case PF_Cmd_ABOUT:
                ERR(About(in_data, out_data, params, output));
            break;

            case PF_Cmd_GLOBAL_SETUP:
                ERR(GlobalSetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_GLOBAL_SETDOWN:
                ERR(GlobalSetdown(in_data, out_data, params, output));
            break;

            case PF_Cmd_PARAMS_SETUP:
                ERR(ParamsSetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_RENDER:
                ERR(Render(in_data, out_data, params, output));
            break;

            // Handling this selector will ensure that the UI will be properly initialized,
            // even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
            break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
            break;

            default:
            break;
        }
    }
    catch (PF_Err &thrown_err)
    {
        err = thrown_err;
    }

    return err;
}