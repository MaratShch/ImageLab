#include <cstring>
#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "ImageLabDenoiseUtils.hpp"
#include "PrSDKAESupport.h"


PF_Err
ImageLabDenoise_SequenceSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    constexpr int32_t maxTwiddleVectorSize = 24;
    CACHE_ALIGN int32_t twiddleX[maxTwiddleVectorSize]{};
    CACHE_ALIGN int32_t twiddleY[maxTwiddleVectorSize]{};

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = in_data->width;
    const A_long sizeY = in_data->height;

    const int sTwiddleX = proc_compute_prime (sizeX, maxTwiddleVectorSize, twiddleX);
    const int sTwiddleY = proc_compute_prime (sizeY, maxTwiddleVectorSize, twiddleY);

    if (0 != sTwiddleX && 0 != sTwiddleY)
    {
        auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
        PF_Handle seqDataHndl = handleSuite->host_new_handle(sizeof(strFftHandle));
        if (nullptr != seqDataHndl)
        {
            strFftHandle* fftHandlerH = reinterpret_cast<strFftHandle*>(handleSuite->host_lock_handle(seqDataHndl));
            if (nullptr != fftHandlerH)
            {
                AEFX_CLR_STRUCT_EX(*fftHandlerH);

                fftHandlerH->isFlat = false;
                fftHandlerH->twiddleStr.sizeX = sizeX;
                fftHandlerH->twiddleStr.sizeY = sizeY;
                fftHandlerH->twiddleStr.twiddleXSize = sTwiddleX;
                fftHandlerH->twiddleStr.twiddleYSize = sTwiddleY;
                fftHandlerH->valid = static_cast<A_long>(sTwiddleX && sTwiddleY);

                std::memcpy(fftHandlerH->twiddleStr.twiddleX, twiddleX, sTwiddleX * sizeof(int32_t));
                std::memcpy(fftHandlerH->twiddleStr.twiddleY, twiddleY, sTwiddleY * sizeof(int32_t));

                if (PremierId != in_data->appl_id)
                {
                    AEFX_SuiteScoper<AEGP_UtilitySuite3> u_suite(in_data, kAEGPUtilitySuite, kAEGPUtilitySuiteVersion3);
                    u_suite->AEGP_RegisterWithAEGP(nullptr, strName, &fftHandlerH->id);
                }

                // notify AE that this is our sequence data handle
                out_data->sequence_data = seqDataHndl;

                // unlock handle
                handleSuite->host_unlock_handle(seqDataHndl);
            } // if (nullptr != seqP)
        } // if (nullptr != seqDataHndl)
        else
            err = PF_Err_OUT_OF_MEMORY;

    } // if (0 != sTwiddleX && 0 != sTwiddleY)
    else
        out_data->sequence_data = nullptr;

    return err;
}


PF_Err
ImageLabDenoise_SequenceReSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    PF_Err err = PF_Err_NONE;

    auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
    PF_Handle seqDataHndl = handleSuite->host_new_handle(sizeof(strFftHandle));

    // if sequence data is present
    if (nullptr != in_data->sequence_data)
    {
        // get handle to flat data ... 
        PF_Handle flatSequenceDataH = in_data->sequence_data;
        // ... then get its actual data pointer/
        strFftHandle* flatSequenceDataP = static_cast<strFftHandle*>(GET_OBJ_FROM_HNDL(flatSequenceDataH));
        if (nullptr != flatSequenceDataP && true == flatSequenceDataP->isFlat)
        {
            // create a new handle, allocating the size of your (unflat) sequence data for it
            PF_Handle unflatSequenceDataH = handleSuite->host_new_handle(sizeof(strFftHandle));
            if (nullptr != unflatSequenceDataH)
            {
                // lock and get actual data pointer for unflat data
                strFftHandle* unflatSequenceDataP = static_cast<strFftHandle*>(handleSuite->host_lock_handle(unflatSequenceDataH));
                if (nullptr != unflatSequenceDataP)
                {
                    AEFX_CLR_STRUCT_EX(*unflatSequenceDataP);
                    
                    // set flag for being "unflat"
                    unflatSequenceDataP->isFlat = false;

                    // directly copy int value unflat -> flat
                    unflatSequenceDataP->id = flatSequenceDataP->id;
                    unflatSequenceDataP->valid = flatSequenceDataP->valid;

                    // copy twiddle structure
                    std::memcpy(&unflatSequenceDataP->twiddleStr, &flatSequenceDataP->twiddleStr, sizeof(unflatSequenceDataP->twiddleStr));
                    
                    // notify AE of unflat sequence data
                    out_data->sequence_data = unflatSequenceDataH;

                    // dispose flat sequence data!
                    handleSuite->host_dispose_handle(flatSequenceDataH);
                    in_data->sequence_data = nullptr;
                } // if (nullptr != unflatSequenceDataP)
                else
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;

                // unlock unflat sequence data handle
                handleSuite->host_unlock_handle(unflatSequenceDataH);

            } // if (nullptr != unflatSequenceDataH)

        } // if (nullptr != flatSequenceDataP && true == flatSequenceDataP->isFlat)
        else
        {
            // use input unflat data as unchanged output
            out_data->sequence_data = in_data->sequence_data;
        }
    } // if (nullptr != in_data->sequence_data)
    else
    {
        // no sequence data exists ? Let's create one!
        err = ImageLabDenoise_SequenceSetup (in_data, out_data, params, output);
    }

    return err;
}


PF_Err
ImageLabDenoise_SequenceFlatten
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    PF_Err err = PF_Err_NONE;

    auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
    if (nullptr != in_data->sequence_data)
    {
        // assume it's always unflat data and get its handle ... /
        PF_Handle unflatSequenceDataH = in_data->sequence_data;
        // ... then get its actual data pointer
        strFftHandle* unflatSequenceDataP = static_cast<strFftHandle*>(GET_OBJ_FROM_HNDL(unflatSequenceDataH));
        if (nullptr != unflatSequenceDataP)
        {
            // create a new handle, allocating the size of our (flat) sequence data for it
            PF_Handle flatSequenceDataH = handleSuite->host_new_handle(sizeof(strFftHandle));
            if (nullptr != flatSequenceDataH)
            {
                // lock and get actual data pointer for flat data
                strFftHandle* flatSequenceDataP = static_cast<strFftHandle*>(handleSuite->host_lock_handle(flatSequenceDataH));
                if (nullptr != flatSequenceDataP)
                {
                    // clear structure fields
                    AEFX_CLR_STRUCT_EX(*flatSequenceDataP);

                    // set flag for being "unflat"
                    unflatSequenceDataP->isFlat = false;

                    // directly copy int value unflat -> flat
                    unflatSequenceDataP->id = flatSequenceDataP->id;
                    unflatSequenceDataP->valid = flatSequenceDataP->valid;

                    // copy twiddle structure
                    std::memcpy(&unflatSequenceDataP->twiddleStr, &flatSequenceDataP->twiddleStr, sizeof(unflatSequenceDataP->twiddleStr));

                    // notify AE of new flat sequence data
                    out_data->sequence_data = flatSequenceDataH;

                    // unlock flat sequence data handle
                    handleSuite->host_unlock_handle(flatSequenceDataH);
                } // if (nullptr != flatSequenceDataP)

            } // if (nullptr != flatSequenceDataH)
            else
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;

            // dispose unflat sequence data! */
            handleSuite->host_dispose_handle(unflatSequenceDataH);
            in_data->sequence_data = nullptr;
        } // if (nullptr != unflatSequenceDataP)

    } // if (nullptr != in_data->sequence_data)
    else
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    return err;
}


PF_Err
ImageLabDenoise_SequenceSetdown
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    if (nullptr != in_data->sequence_data)
    {
        auto const handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        handleSuite->host_dispose_handle (in_data->sequence_data);
        in_data->sequence_data = nullptr;
    }

    // Invalidate the sequence_data pointers in both AE's input and output data fields (to signal that we have properly disposed of the data)
    out_data->sequence_data = nullptr;

    return PF_Err_NONE;
}
