#include "AlgoRules.hpp"
#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureAlgo.hpp"
#include "ColorTemperatureDraw.hpp"
#include "ColorTemperatureControlsPresets.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageLabMemInterface.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
	PF_Err err{ PF_Err_OUT_OF_MEMORY };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    // --- Acquire controls values --- //
    const strControlSet cctSetup = GetCctSetup (params);
    const AlgoProcT targetCct = cctSetup.Cct;
    const AlgoProcT targetDuv = cctSetup.Duv;
    const eCOLOR_OBSERVER observer = static_cast<eCOLOR_OBSERVER>(cctSetup.observer);
    const eCctType cctValueType = static_cast<eCctType>(cctSetup.cctType);

    if (0.f == targetCct && 0.f == targetDuv)
        return PF_COPY(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld, output, NULL, NULL);

    // This plugin called frop PR - check video fomat
    auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

    const pHandle* pStr = static_cast<const pHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));

    if ((nullptr != pStr) && PF_Err_NONE == (err = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
    {
        AlgoCCT::CctHandleF32* cctHandle = pStr->hndl;
        if (nullptr != cctHandle)
        {
            const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
            const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
            const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;

            // Allocate memory storage for store temporary results
            const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr<AlgoProcT>)), CACHE_LINE);

            void* pMemoryBlock = nullptr;
            A_long blockId = ::GetMemoryBlock (totalProcMem, 0, &pMemoryBlock);

            if (nullptr != pMemoryBlock && blockId >= 0)
            {
                PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT> *__restrict> (pMemoryBlock);

                switch (destinationPixelFormat)
                {
                    case PrPixelFormat_BGRA_4444_8u:
                    {
                        const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                              PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u8_value_white);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                        const AdaptationMatrixT matrix = computeAdaptationMatrix (cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct (localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(u8_value_white));

                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    case PrPixelFormat_BGRA_4444_16u:
                    {
                        const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                              PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u16_value_white);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);
  
                        AdaptationMatrixT matrix = computeAdaptationMatrix(cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct(localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(u16_value_white));

                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    case PrPixelFormat_BGRA_4444_32f:
                    {
                        const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                              PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                        AdaptationMatrixT matrix = computeAdaptationMatrix(cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct(localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(1));
                        
                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    case PrPixelFormat_VUYA_4444_8u_709:
                    case PrPixelFormat_VUYA_4444_8u:
                    {
                        const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                              PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u8_value_white);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                        AdaptationMatrixT matrix = computeAdaptationMatrix(cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct(localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(u8_value_white));
                        
                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    case PrPixelFormat_VUYA_4444_32f_709:
                    case PrPixelFormat_VUYA_4444_32f:
                    {
                        const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                              PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                        AdaptationMatrixT matrix = computeAdaptationMatrix(cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct(localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(1));

                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    case PrPixelFormat_RGB_444_10u:
                    {
                        const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                              PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                        const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                        constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u10_value_white);
                        const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                        const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                        AdaptationMatrixT matrix = computeAdaptationMatrix(cctHandle, observer, cctValueType, cct_duv, std::make_pair(targetCct, targetDuv));

                        AdjustCct(localSrc, localDst, matrix, sizeX, sizeY, linePitch, linePitch, static_cast<AlgoProcT>(u10_value_white));
                        
                        // Draw CCT/Duv values on Effect Panel
                        SetGUI_CCT(cct_duv);
                    }
                    break;

                    default:
                        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                    break;
                } // switch (destinationPixelFormat)

                ::FreeMemoryBlock(blockId);
                blockId = -1;
                pMemoryBlock = nullptr;

                err = PF_Err_NONE;

            } // if (nullptr != pMemoryBlock && blockId >= 0)


        } // if (nullptr != cctHandle)

    } // if (nullptr != pStr && PF_Err_NONE == (err = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))

	return err;
}
