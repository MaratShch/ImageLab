#include "ImageLabSketch.h"

template <typename T>
void process_RGB_buffer
(
	const T*  __restrict  pSrc,
	T*  __restrict  pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	AlgMemStorage*        pMemDesc
)
{
	/* temporary buffer for vertial gradient */
	float* __restrict p1 = reinterpret_cast<float* __restrict>(pMemDesc->pBuf1);
	/* temporary buffer for horizontal gradient */
	float* __restrict p2 = reinterpret_cast<float* __restrict>(pMemDesc->pBuf2);

	const bool isBT709 = width > 720 ? true : false;

	ImageGradientVertical_RGB  (pSrc, p1, width, height, linePitch, isBT709);
	ImageGradientHorizontal_RGB(pSrc, p2, width, height, linePitch, isBT709);

	return;
}


template <typename T>
void process_YUV_buffer
(
	const T*  __restrict  pSrc,
	      T*  __restrict  pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	AlgMemStorage*        pMemDesc
)
{
	/* temporary buffer for vertial gradient */
	float* __restrict p1 = reinterpret_cast<float* __restrict>(pMemDesc->pBuf1);
	/* temporary buffer for horizontal gradient */
	float* __restrict p2 = reinterpret_cast<float* __restrict>(pMemDesc->pBuf2);

	ImageGradientVertical_YUV  (pSrc, p1, width, height, linePitch);
	ImageGradientHorizontal_YUV(pSrc, p2, width, height, linePitch);

	return;
}


csSDK_int32 selectProcessFunction (VideoHandle theData)
{
	static constexpr char* strPpixSuite { "Premiere PPix Suite" };
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH	       paramsH = nullptr;
	AlgMemStorage*         pMemStorage = nullptr;
	constexpr long         siteVersion = 1;
	csSDK_int32            errCode = fsBadFormatIndex;
	csSDK_int8             isCharcoalSketch = 0;
	csSDK_int8             isImageEnhancement = 0;
	bool                   processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	PPixSuite = nullptr;
		const SPErr     err = SPBasic->AcquireSuite (strPpixSuite, siteVersion, (const void**)&PPixSuite);

		if (kSPNoError == err && nullptr != PPixSuite)
		{
			// Get the frame dimensions
			prRect box = {};
			((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

			// Calculate dimensions
			const csSDK_int32 height = box.bottom - box.top;
			const csSDK_int32 width  = box.right - box.left;
			const csSDK_int32 linePitch = (((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination)) >> 2;

			// Check is frame dimensions are correct
			if (0 >= height || 0 >= width || 0 >= linePitch || linePitch < width)
				return fsBadFormatIndex;

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;

			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr != paramsH)
			{
				isCharcoalSketch   = (*paramsH)->checkbox1;
				isImageEnhancement = (*paramsH)->checkbox2;

				/* check is temporary memory available */
				if (nullptr != (pMemStorage = (*paramsH)->pAlgMem))
				{
					/* check and realloc available memory size */
					const size_t tmpMemSize = pMemStorage->bytesSize;
					const size_t frameSize = width * height * sizeof(float);
					if (tmpMemSize < frameSize)
					{
						/* need more memory - realloc memory with new size */
						if (false == algMemStorageRealloc(width, height, pMemStorage))
							return fsBadFormatIndex;
					}
				}
				else
					return fsBadFormatIndex;
			}

			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			prSuiteError  siteErr = suiteError_Fail;
			if (suiteError_NoError != (siteErr = PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat)))
				return fsBadFormatIndex;

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const PixelBGRA_u8* __restrict pSrcPix = reinterpret_cast<const PixelBGRA_u8* __restrict>(srcImg);
					PixelBGRA_u8* __restrict pDstPix = reinterpret_cast<PixelBGRA_u8* __restrict>(dstImg);

					process_RGB_buffer (pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				{
					const PixelBGRA_u16* __restrict pSrcPix = reinterpret_cast<const PixelBGRA_u16* __restrict>(srcImg);
					      PixelBGRA_u16* __restrict pDstPix = reinterpret_cast<PixelBGRA_u16* __restrict>(dstImg);

					process_RGB_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const PixelBGRA_f32* __restrict pSrcPix = reinterpret_cast<const PixelBGRA_f32* __restrict>(srcImg);
					      PixelBGRA_f32* __restrict pDstPix = reinterpret_cast<PixelBGRA_f32* __restrict>(dstImg);

					process_RGB_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				{
					const PixelYUVA_u8* __restrict pSrcPix = reinterpret_cast<const PixelYUVA_u8* __restrict>(srcImg);
					      PixelYUVA_u8* __restrict pDstPix = reinterpret_cast<PixelYUVA_u8* __restrict>(dstImg);

					process_YUV_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				{
					const PixelYUVA_f32* __restrict pSrcPix = reinterpret_cast<const PixelYUVA_f32* __restrict>(srcImg);
					      PixelYUVA_f32* __restrict pDstPix = reinterpret_cast<PixelYUVA_f32* __restrict>(dstImg);

					process_YUV_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				{
					const PixelARGB_u8* __restrict pSrcPix = reinterpret_cast<const PixelARGB_u8* __restrict>(srcImg);
					      PixelARGB_u8* __restrict pDstPix = reinterpret_cast<PixelARGB_u8* __restrict>(dstImg);

					process_RGB_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_ARGB_4444_16u:
				{
					const PixelARGB_u16* __restrict pSrcPix = reinterpret_cast<const PixelARGB_u16* __restrict>(srcImg);
					      PixelARGB_u16* __restrict pDstPix = reinterpret_cast<PixelARGB_u16* __restrict>(dstImg);

					process_RGB_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				case PrPixelFormat_ARGB_4444_32f:
				{
					const PixelARGB_f32* __restrict pSrcPix = reinterpret_cast<const PixelARGB_f32* __restrict>(srcImg);
					      PixelARGB_f32* __restrict pDstPix = reinterpret_cast<PixelARGB_f32* __restrict>(dstImg);

					process_RGB_buffer(pSrcPix, pDstPix, width, height, linePitch, pMemStorage);
				}
				break;

				// =========== miscellanous formats =========================== //
//				case PrPixelFormat_RGB_444_10u:
//					processSucceed = processSepiaRGB444_10u_slice(theData);
//				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite (strPpixSuite, siteVersion);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData)
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	filterParamsH		paramsH = nullptr;
	csSDK_int32			errCode = fsNoErr;
	constexpr csSDK_uint32	filterParamSize = sizeof(filterParams);

	switch (selector)
	{
		case fsInitSpec:
			if ((*theData)->specsHandle)
			{
				// In a filter that has a need for a more complex setup dialog
				// you would present your platform specific user interface here,
				// storing results in the specsHandle (which you've allocated).
			}
			else
			{
				paramsH = reinterpret_cast<filterParamsH>(((*theData)->piSuites->memFuncs->newHandle)(filterParamSize));

				if (nullptr != paramsH)
				{
					(*paramsH)->checkbox1 = 0;
					(*paramsH)->checkbox2 = 0;
					(*paramsH)->pAlgMem = getAlgStorageStruct();
				}

				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
		break;


		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
		break;

		case fsDisposeData:
			if (nullptr != (*theData)->specsHandle)
			{
				/* not free! alg memory storage here! AlgMemoryStorage will be freed on DLL_PROCESS_DETACH */
				(*theData)->piSuites->memFuncs->disposeHandle((*theData)->specsHandle);
				(*theData)->specsHandle = nullptr;
			}
		break;

		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;
			
		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported(theData);
		break;

		case fsCacheOnLoad:
			errCode = fsDoNotCacheOnLoad;
		break;

		default:
			// unhandled case
		break;
		
	}

	return errCode;
}
