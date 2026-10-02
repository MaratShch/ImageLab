#pragma once

#include <atomic>
#include <tuple>
#include <sstream>
#include <cuda_runtime.h>
#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKGPUImageProcessingSuite.h"
#include "PrSDKGPUFilter.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPPix2Suite.h"
#include "PrSDKVideoSegmentSuite.h"
#include "ClassRestrictions.hpp"


#ifdef __cplusplus
#define PLUGIN_GPU_ENTRY_POINT_CALL	extern "C" DllExport
#else
#define PLUGIN_GPU_ENTRY_POINT_CALL DllExport
#endif

constexpr int32_t ImageLabGpuPixel16f = PrPixelFormat_GPU_BGRA_4444_16f;
constexpr int32_t ImageLabGpuPixel32f = PrPixelFormat_GPU_BGRA_4444_32f;


class CImageLab2GpuObj
{
public:

	CLASS_NON_COPYABLE(CImageLab2GpuObj);
	CLASS_NON_MOVABLE(CImageLab2GpuObj);

	CImageLab2GpuObj();
	virtual ~CImageLab2GpuObj();

	static prSuiteError Startup (piSuitesPtr piSuites, csSDK_int32 inIndex)
	{
		return suiteError_NoError;
	}

	static prSuiteError Shutdown (piSuitesPtr piSuites, csSDK_int32 inIndex)
	{
		return suiteError_NoError;
	}

	static csSDK_int32 PluginCount()
	{
		return 1;
	}

	static PrSDKString MatchName (piSuitesPtr piSuites, csSDK_int32 inIndex)
	{
		return PrSDKString();
	}

	static const csSDK_uint32 TotalInstances(void)
	{
		const csSDK_uint32 cnt = objCnt;
		return cnt;
	}

	virtual prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData);
	virtual prSuiteError Cleanup (void);

	virtual prSuiteError GetFrameDependencies(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32* ioQueryIndex, PrGPUFilterFrameDependency* outFrameRequirements);

	virtual prSuiteError Precompute(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32 inIndex, PPixHand inFrame);

	virtual prSuiteError Render(const PrGPUFilterRenderParams* inRenderParams, const PPixHand* inFrames, csSDK_size_t inFrameCount, PPixHand* outFrame) = 0;


protected:
	template<typename T>
	prSuiteError GetProperty(
		const char* inKey,
		T& outValue)
	{
		PrMemoryPtr buffer = nullptr;
		prSuiteError suiteError = mVideoSegmentSuite->GetNodeProperty(mNodeID, inKey, &buffer);
		if (PrSuiteErrorSucceeded(suiteError))
		{
			std::istringstream stream((const char*)buffer);
			stream >> outValue;
			mMemoryManagerSuite->PrDisposePtr(buffer);
		}
		return suiteError;
	}

    // Helper to pad sizes to the nearest 256-byte boundary
    const inline size_t AlignSizeCuda (size_t size) noexcept
    {
        // CUDA aligns memory to 256 bytes for optimal warp memory transactions
        constexpr size_t CUDA_ALIGNMENT = 256ull;
        return (size + CUDA_ALIGNMENT - 1) & ~(CUDA_ALIGNMENT - 1);
    }

    const std::tuple<size_t, size_t> GetGpuMemoryInfo_CUDA(void) noexcept
    {
        size_t free_byte = 0ull, total_byte = 0ull;

        // Bind to the device Premiere assigned to this instance before querying.
        if (cudaSuccess != cudaSetDevice(static_cast<int>(mDeviceIndex)) || cudaSuccess != cudaMemGetInfo(&free_byte, &total_byte))
            return std::make_tuple(size_t{ 0 }, size_t{ 0 });
        
        return std::make_tuple(free_byte, total_byte);
    }

    inline const size_t GetSafeMargin_CUDA (void) noexcept
    {
        constexpr size_t safeMargin = static_cast<size_t>(128 * 1024 * 1024);
        return safeMargin;
    }


	PrParam GetParam (csSDK_int32 inIndex, PrTime inTime);
	const size_t RoundUp (size_t inValue, size_t inMultiple);

	const int GetGPUBytesPerPixel (const PrPixelFormat inPixelFormat);

	SPBasicSuite* mBasicSite;
	PrSDKGPUDeviceSuite* mGPUDeviceSuite;
	PrSDKGPUImageProcessingSuite* mGPUImageProcessingSuite;
	PrSDKMemoryManagerSuite* mMemoryManagerSuite;
	PrSDKPPixSuite* mPPixSuite;
	PrSDKPPix2Suite* mPPix2Suite;
	PrSDKVideoSegmentSuite* mVideoSegmentSuite;

	piSuitesPtr mSuites;
	PrTimelineID mTimelineID;
	csSDK_int32 mNodeID;
	csSDK_uint32 mDeviceIndex;
	PrGPUDeviceInfo mDeviceInfo;

private:
	static std::atomic<uint32_t>objCnt;
};

/**
**
*/
template<class GPUFilter>
	struct PrGPUFilterModule
{
	static prSuiteError Startup(
		piSuitesPtr piSuites,
		csSDK_int32* ioIndex,
		PrGPUFilterInfo* outFilterInfo)
	{
		csSDK_int32 index = *ioIndex;
		if (index + 1 > GPUFilter::PluginCount())
		{
			return suiteError_InvalidParms;
		}
		if (index + 1 < GPUFilter::PluginCount())
		{
			*ioIndex += 1;
		}

		outFilterInfo->outMatchName = GPUFilter::MatchName(piSuites, index);
		outFilterInfo->outInterfaceVersion = PrSDKGPUFilterInterfaceVersion;

		return GPUFilter::Startup(piSuites, *ioIndex);
	}

	static prSuiteError Shutdown(
		piSuitesPtr piSuites,
		csSDK_int32* ioIndex)
	{
		return GPUFilter::Shutdown(piSuites, *ioIndex);
	}

    static prSuiteError CreateInstance (PrGPUFilterInstance* ioInstanceData)
    {
        if (nullptr == ioInstanceData)
            return suiteError_InvalidParms;

        GPUFilter* gpuFilter = nullptr;
        
        try { gpuFilter = new GPUFilter(); }
        catch (...) { return suiteError_OutOfMemory; }
        
        const prSuiteError result = gpuFilter->Initialize(ioInstanceData);
        if (PrSuiteErrorSucceeded(result))
            ioInstanceData->ioPrivatePluginData = gpuFilter;
        else
            delete gpuFilter;

        return result;
    }

	static prSuiteError DisposeInstance(
		PrGPUFilterInstance* ioInstanceData)
	{
        GPUFilter* filter = static_cast<GPUFilter*>(ioInstanceData->ioPrivatePluginData);
        if (nullptr != filter)
        {
            delete filter;
            ioInstanceData->ioPrivatePluginData = nullptr;
            filter = nullptr;
        }
        return suiteError_NoError;
	}

	static prSuiteError GetFrameDependencies(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		csSDK_int32* ioQueryIndex,
		PrGPUFilterFrameDependency* outFrameRequirements)
	{
        GPUFilter* filter = static_cast<GPUFilter*>(inInstanceData->ioPrivatePluginData);
        return (nullptr != filter) ? filter->GetFrameDependencies(inRenderParams, ioQueryIndex, outFrameRequirements) : suiteError_InvalidParms;
	}

	static prSuiteError Precompute(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		csSDK_int32 inIndex,
		PPixHand inFrame)
	{
        GPUFilter* filter = static_cast<GPUFilter*>(inInstanceData->ioPrivatePluginData);
		return (nullptr != filter) ? filter->Precompute(inRenderParams, inIndex, inFrame) : suiteError_InvalidParms;
	}

	static prSuiteError Render(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{
        GPUFilter* filter = static_cast<GPUFilter*>(inInstanceData->ioPrivatePluginData);
		return (nullptr != filter) ? filter->Render(inRenderParams, inFrames, inFrameCount, outFrame) : suiteError_InvalidParms;
	}
};


#ifndef DECLARE_GPUFILTER_ENTRY
#define DECLARE_GPUFILTER_ENTRY(ClassName) \
	PLUGIN_GPU_ENTRY_POINT_CALL prSuiteError xGPUFilterEntry( \
	csSDK_uint32 inHostInterfaceVersion, \
	csSDK_int32* ioIndex, \
	prBool inStartup, \
	piSuitesPtr piSuites, \
	PrGPUFilter* outFilter, \
	PrGPUFilterInfo* outFilterInfo) \
	{ \
		if (inStartup) \
		{ \
			outFilter->CreateInstance = ClassName::CreateInstance; \
			outFilter->DisposeInstance = ClassName::DisposeInstance; \
			outFilter->GetFrameDependencies = ClassName::GetFrameDependencies; \
			outFilter->Precompute = ClassName::Precompute; \
			outFilter->Render = ClassName::Render; \
			return ClassName::Startup(piSuites, ioIndex, outFilterInfo); \
		} \
		else \
		{ \
			return ClassName::Shutdown(piSuites, ioIndex); \
		} \
	}

#endif