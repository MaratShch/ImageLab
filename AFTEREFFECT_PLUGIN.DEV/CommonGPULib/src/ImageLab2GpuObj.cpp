#include "ImageLab2GpuObj.hpp"

std::atomic<uint32_t>CImageLab2GpuObj::objCnt = 0u;

CImageLab2GpuObj::CImageLab2GpuObj()
{
	objCnt++;
	mGPUDeviceSuite = nullptr;
	mGPUImageProcessingSuite = nullptr;
	mMemoryManagerSuite = nullptr;
	mPPixSuite = nullptr;
	mPPix2Suite = nullptr;
	mVideoSegmentSuite = nullptr;
	mSuites = nullptr;

	mTimelineID = -1;
	mNodeID = -1;
	mDeviceIndex = 0;
	memset (&mDeviceInfo, 0, sizeof(mDeviceInfo));

	return;
}

CImageLab2GpuObj::~CImageLab2GpuObj()
{
	objCnt--;
	return;
}


prSuiteError CImageLab2GpuObj::Initialize (PrGPUFilterInstance* ioInstanceData)
{
	prSuiteError err = suiteError_IDNotValid;

	if (nullptr != ioInstanceData)
	{
		const piSuitesPtr mSuites = ioInstanceData->piSuites;

		if (nullptr != mSuites)
		{
			const PlugUtilFuncsPtr pUtilFuncs = mSuites->utilFuncs;

			if (nullptr != pUtilFuncs)
			{
				mBasicSite = pUtilFuncs->getSPBasicSuite();

				if (nullptr != mBasicSite)
				{
					const SPErr errGpuDev = 
						mBasicSite->AcquireSuite (kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion, (const void**)&mGPUDeviceSuite);

					const SPErr errGpuPrc = 
						mBasicSite->AcquireSuite (kPrSDKGPUImageProcessingSuite, kPrSDKGPUImageProcessingSuiteVersion, (const void**)&mGPUImageProcessingSuite);

					const SPErr errMemMng = 
						mBasicSite->AcquireSuite (kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion, (const void**)&mMemoryManagerSuite);

					const SPErr errPpxSit =
						mBasicSite->AcquireSuite (kPrSDKPPixSuite, kPrSDKPPixSuiteVersion, (const void**)&mPPixSuite);

					const SPErr errPpx2Sit =
						mBasicSite->AcquireSuite (kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion, (const void**)&mPPix2Suite);

					const SPErr errVidSeg =
						mBasicSite->AcquireSuite(kPrSDKVideoSegmentSuite, kPrSDKVideoSegmentSuiteVersion, (const void**)&mVideoSegmentSuite);

					mTimelineID = ioInstanceData->inTimelineID;
					mNodeID = ioInstanceData->inNodeID;
					mDeviceIndex = ioInstanceData->inDeviceIndex;

					mGPUDeviceSuite->GetDeviceInfo(kPrSDKGPUDeviceSuiteVersion, mDeviceIndex, &mDeviceInfo);

					/* start sites validation */
					if (suiteError_NoError == errGpuDev && suiteError_NoError == errGpuPrc  && suiteError_NoError == errMemMng &&
						suiteError_NoError == errPpxSit && suiteError_NoError == errPpx2Sit && suiteError_NoError == errVidSeg)
					{
						err = suiteError_NoError;
					}
				}
			}
		}

	}
	return err;
}


prSuiteError CImageLab2GpuObj::Cleanup(void)
{
	if (nullptr != mBasicSite)
	{
		mGPUDeviceSuite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion);

		mGPUImageProcessingSuite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKGPUImageProcessingSuite, kPrSDKGPUImageProcessingSuiteVersion);

		mMemoryManagerSuite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion);

		mPPixSuite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKPPixSuite, kPrSDKPPixSuiteVersion);

		mPPix2Suite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion);

		mVideoSegmentSuite = nullptr;
		mBasicSite->ReleaseSuite (kPrSDKVideoSegmentSuite, kPrSDKVideoSegmentSuiteVersion);

		memset(&mDeviceInfo, 0, sizeof(mDeviceInfo));

		mBasicSite = nullptr;
	}

	return suiteError_NoError;
}


prSuiteError CImageLab2GpuObj::GetFrameDependencies(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32* ioQueryIndex, PrGPUFilterFrameDependency* outFrameRequirements)
{
	return suiteError_NotImplemented;
}

prSuiteError CImageLab2GpuObj::Precompute(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32 inIndex, PPixHand inFrame)
{
	return suiteError_NotImplemented;
}

const int CImageLab2GpuObj::GetGPUBytesPerPixel (const PrPixelFormat inPixelFormat)
{
	return (PrPixelFormat_GPU_BGRA_4444_32f == inPixelFormat ? 16 : 8);
}


PrParam CImageLab2GpuObj::GetParam (csSDK_int32 inIndex, PrTime inTime)
{
	PrParam param;
	inIndex -= 1; // GPU filters do not include the input frame
	mVideoSegmentSuite->GetParam (mNodeID, inIndex, inTime, &param);
	return param;
}

const size_t CImageLab2GpuObj::RoundUp (size_t inValue, size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}
