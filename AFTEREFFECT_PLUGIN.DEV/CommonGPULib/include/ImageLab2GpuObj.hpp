#pragma once

#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKGPUImageProcessingSuite.h"
#include "PrSDKGPUFilter.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPPix2Suite.h"
#include "PrSDKVideoSegmentSuite.h"
#include "ClassRestrictions.hpp"
#include <atomic>

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
		PrMemoryPtr buffer;
		prSuiteError suiteError = mVideoSegmentSuite->GetNodeProperty(mNodeID, inKey, &buffer);
		if (PrSuiteErrorSucceeded(suiteError))
		{
			std::istringstream stream((const char*)buffer);
			stream >> outValue;
			mMemoryManagerSuite->PrDisposePtr(buffer);
		}
		return suiteError;
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

	static prSuiteError CreateInstance(
		PrGPUFilterInstance* ioInstanceData)
	{
		GPUFilter* gpuFilter = new GPUFilter();
		prSuiteError result = gpuFilter->Initialize(ioInstanceData);
		if (PrSuiteErrorSucceeded(result))
		{
			ioInstanceData->ioPrivatePluginData = gpuFilter;
		}
		else
		{
			delete gpuFilter;
		}
		return result;
	}

	static prSuiteError DisposeInstance(
		PrGPUFilterInstance* ioInstanceData)
	{
		delete (GPUFilter*)ioInstanceData->ioPrivatePluginData;
		ioInstanceData->ioPrivatePluginData = 0;
		return suiteError_NoError;
	}

	static prSuiteError GetFrameDependencies(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		csSDK_int32* ioQueryIndex,
		PrGPUFilterFrameDependency* outFrameRequirements)
	{
		return ((GPUFilter*)inInstanceData->ioPrivatePluginData)->GetFrameDependencies(inRenderParams, ioQueryIndex, outFrameRequirements);
	}

	static prSuiteError Precompute(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		csSDK_int32 inIndex,
		PPixHand inFrame)
	{
		return ((GPUFilter*)inInstanceData->ioPrivatePluginData)->Precompute(inRenderParams, inIndex, inFrame);
	}

	static prSuiteError Render(
		PrGPUFilterInstance* inInstanceData,
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{
		return ((GPUFilter*)inInstanceData->ioPrivatePluginData)->Render(inRenderParams, inFrames, inFrameCount, outFrame);
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