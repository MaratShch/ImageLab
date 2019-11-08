#include "AdobeImageLabAWB.h"


CACHE_ALIGN constexpr double sRGBtoXYZ[9] = {
	0.4124564,  0.3575761,  0.1804375,
	0.2126729,  0.7151522,  0.0721750,
	0.0193339,  0.1191920,  0.9503041
};

CACHE_ALIGN constexpr double XYZtosRGB[9] = {
	3.240455, -1.537139, -0.498532,
   -0.969266,  1.876011,  0.041556,
    0.055643, -0.204026,  1.057225
};

CACHE_ALIGN constexpr double CAT2[9] = {
	0.73280, 0.42960,  -0.16240,
   -0.70360, 1.69750,   0.00610,
    0.00300, 0.01360,   0.98340
};

CACHE_ALIGN constexpr double invCAT2[9] = {
	1.09610, -0.27890, 0.18270,
	0.45440,  0.47350, 0.07210,
   -0.00960, -0.00570, 1.01530
};


CACHE_ALIGN constexpr double yuv2xyz[3] = { 0.114653800, 0.083911980, 0.082220770 };
CACHE_ALIGN constexpr double xyz2yuv[3] = { 0.083911980, 0.283096500, 0.466178900 };


const double* const GetIlluminate(const eILLIUMINATE illuminateIdx)
{
	CACHE_ALIGN static constexpr double tblIlluminate[12][3] = {
		{  95.0470, 100.0000, 108.8830 }, // DAYLIGHT - D65 (DEFAULT)
		{  98.0740, 100.0000, 118.2320 }, // OLD_DAYLIGHT
		{  99.0927, 100.0000,  85.3130 }, // OLD_DIRECT_SUNLIGHT_AT_NOON
		{  95.6820, 100.0000,  92.1490 }, // MID_MORNING_DAYLIGHT
		{  94.9720, 100.0000, 122.6380 }, // NORTH_SKY_DAYLIGHT
		{  92.8340, 100.0000, 103.6650 }, // DAYLIGHT_FLUORESCENT_F1
		{  99.1870, 100.0000,  67.3950 }, // COOL_FLUERESCENT
		{ 103.7540, 100.0000,  49.8610 }, // WHITE_FLUORESCENT
		{ 109.1470, 100.0000,  38.8130 }, // WARM_WHITE_FLUORESCENT
		{  90.8720, 100.0000,  98.7230 }, // DAYLIGHT_FLUORESCENT_F5
		{ 100.3650, 100.0000,  67.8680 }  // COOL_WHITE_FLUORESCENT
	};

	return tblIlluminate[illuminateIdx];
}



// in future split this function for more little API's
bool procesBGRA4444_8u_slice(	VideoHandle theData, 
								const double* __restrict pMatrixIn,
								const double* __restrict pMatrixOut)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	double U_avg[maxIterCount] = {};
	double V_avg[maxIterCount] = {};

	prRect box = { 0 };

	// Get the filter parameters
	const FilterParamHandle filterParamH = reinterpret_cast<FilterParamHandle>((*theData)->specsHandle);
	// get iteration count and gray threshold from sliders
	const csSDK_int32 iterCnt		= (nullptr != filterParamH) ? (*filterParamH)->sliderIterCnt : 1;
	const csSDK_int32 grayThr		= (nullptr != filterParamH) ? (*filterParamH)->sliderGrayThr : 30;
	const eILLIUMINATE setIlluminate= (nullptr != filterParamH) ? (*filterParamH)->illuminate : DAYLIGHT;

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	int totalGray;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	const csSDK_uint32* __restrict dstFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const int lastIter = iterCnt - 1;

	double R, G, B;
	double Y, U, V;
	double xX, xY, xZ;
	double F;
	double U_diff, V_diff, normVal;
	double U_bar, V_bar;
	const double T = get_gray_threshold(grayThr);
	constexpr double b = 0.0010; // convergence threshold
	constexpr double algEpsilon = 1.e-06;

	for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)
	{
		csSDK_uint32* srcImg = const_cast<csSDK_uint32*>(srcFrame);
		// only last iteration going with double buffering
		csSDK_uint32* dstImg = const_cast<csSDK_uint32*>((lastIter == iter_cnt) ? dstFrame : srcFrame);

		U_bar = V_bar = 0.0;
		totalGray = 0;

		// first pass - accquire color statistics
		for (int j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
				for (int i = 0; i < width; i++)
				{
					const csSDK_uint32 BGRAPixel = *srcImg;
					srcImg++;

					R = static_cast<double>((BGRAPixel & 0x00FF0000) >> 16);
					G = static_cast<double>((BGRAPixel & 0x0000FF00) >> 8);
					B = static_cast<double>( BGRAPixel & 0x000000FF);

					Y = R * pMatrixIn[0] +
						G * pMatrixIn[1] +
						B * pMatrixIn[2];

					U = R * pMatrixIn[3] +
						G * pMatrixIn[4] +
						B * pMatrixIn[5];

					V = R * pMatrixIn[6] +
						G * pMatrixIn[7] +
						B * pMatrixIn[8];

					F = (abs(U) + abs(V)) / Y;

					if (F < T) {
						totalGray++;
						U_bar += U;
						V_bar += V;
					}
				}

			srcImg += linePitch - width;
		}

		U_avg[iter_cnt] = U_bar / totalGray;
		V_avg[iter_cnt] = V_bar / totalGray;

		if (MAX(abs(U_avg[iter_cnt]), abs(V_avg[iter_cnt])) < b)
			return true; // converged

		if (iter_cnt >= 1)
		{
			U_diff = U_avg[iter_cnt] - U_avg[iter_cnt - 1];
			V_diff = V_avg[iter_cnt] - V_avg[iter_cnt - 1];

			normVal = sqrt(U_diff * U_diff + V_diff * V_diff);

			if (normVal < algEpsilon) 
			{
				// U and V no longer improving, so just copy source to destination and break the loop
				copy_src2dst(const_cast<csSDK_uint32*>(srcFrame), const_cast<csSDK_uint32*>(dstFrame), height, width, rowbytes);
				return true; // U and V no longer improving
			}
		}

		// convert the average GRAY from YUV to RGB
		const double restored_R = 100.00 * pMatrixOut[0] + U_avg[iter_cnt] * pMatrixOut[1] + V_avg[iter_cnt] * pMatrixOut[2];
		const double restored_G = 100.00 * pMatrixOut[3] + U_avg[iter_cnt] * pMatrixOut[4] + V_avg[iter_cnt] * pMatrixOut[5];
		const double restored_B = 100.00 * pMatrixOut[6] + U_avg[iter_cnt] * pMatrixOut[7] + V_avg[iter_cnt] * pMatrixOut[8];

		// calculate xy chromaticity
		xX = restored_R * sRGBtoXYZ[0] + restored_G * sRGBtoXYZ[1] + restored_B * sRGBtoXYZ[2];
		xY = restored_R * sRGBtoXYZ[3] + restored_G * sRGBtoXYZ[4] + restored_B * sRGBtoXYZ[5];
		xZ = restored_R * sRGBtoXYZ[6] + restored_G * sRGBtoXYZ[7] + restored_B * sRGBtoXYZ[8];

		const double xXYZSum = xX + xY + xZ;
		const double xyEst[2] = { xX / xXYZSum, xY / xXYZSum };
		const double xyEstDiv = 100.00 / xyEst[1];

		// Converts xyY chromaticity to CIE XYZ.
		const double xyzEst[3] = {
			xyEstDiv * xyEst[0],
			100.00,
			xyEstDiv * (1.00 - xyEst[0] - xyEst[1])
		};

		// get illuminate
		const double* illuminate = GetIlluminate(setIlluminate);

		const double gainTarget[3] = 
		{
			illuminate[0] * CAT2[0] + illuminate[1] * CAT2[1] + illuminate[2] * CAT2[2],
			illuminate[0] * CAT2[3] + illuminate[1] * CAT2[4] + illuminate[2] * CAT2[5],
			illuminate[0] * CAT2[6] + illuminate[1] * CAT2[7] + illuminate[2] * CAT2[8]
		};

		const double gainEstimated[3] =
		{
			xyzEst[0] * CAT2[0] + xyzEst[1] * CAT2[1] + xyzEst[2] * CAT2[2],
			xyzEst[0] * CAT2[3] + xyzEst[1] * CAT2[4] + xyzEst[2] * CAT2[5],
			xyzEst[0] * CAT2[6] + xyzEst[1] * CAT2[7] + xyzEst[2] * CAT2[8]
		};

		// gain = (xfm*xyz_target)./(xfm*xyz_est);
		const double finalGain[3] =
		{
			gainTarget[0] / gainEstimated[0],
			gainTarget[1] / gainEstimated[1],
			gainTarget[2] / gainEstimated[2]
		};

		const double diagGain[9] =
		{
			finalGain[0], 0.0, 0.0 ,
			0.0, finalGain[1], 0.0,
			0.0, 0.0, finalGain[2]
		};

		const double mulGain[9] = // in future must be enclosed in C++ class Matrix: diagGain * CAT2
		{
			diagGain[0] * CAT2[0] + diagGain[1] * CAT2[3] + diagGain[2] * CAT2[6],
			diagGain[0] * CAT2[1] + diagGain[1] * CAT2[4] + diagGain[2] * CAT2[7],
			diagGain[0] * CAT2[2] + diagGain[1] * CAT2[5] + diagGain[2] * CAT2[8],

			diagGain[3] * CAT2[0] + diagGain[4] * CAT2[3] + diagGain[5] * CAT2[6],
			diagGain[3] * CAT2[1] + diagGain[4] * CAT2[4] + diagGain[5] * CAT2[7],
			diagGain[3] * CAT2[2] + diagGain[4] * CAT2[5] + diagGain[5] * CAT2[8],

			diagGain[6] * CAT2[0] + diagGain[7] * CAT2[3] + diagGain[8] * CAT2[6],
			diagGain[6] * CAT2[1] + diagGain[7] * CAT2[4] + diagGain[8] * CAT2[7],
			diagGain[6] * CAT2[2] + diagGain[7] * CAT2[5] + diagGain[8] * CAT2[8]
		};

		const double outMatrix[9] = // in future must be enclosed in C++ class Matrix: invCAT2 * mulGain
		{
			invCAT2[0] * mulGain[0] + invCAT2[1] * mulGain[3] + invCAT2[2] * mulGain[6],
			invCAT2[0] * mulGain[1] + invCAT2[1] * mulGain[4] + invCAT2[2] * mulGain[7],
			invCAT2[0] * mulGain[2] + invCAT2[1] * mulGain[5] + invCAT2[2] * mulGain[8],

			invCAT2[3] * mulGain[0] + invCAT2[4] * mulGain[3] + invCAT2[5] * mulGain[6],
			invCAT2[3] * mulGain[1] + invCAT2[4] * mulGain[4] + invCAT2[5] * mulGain[7],
			invCAT2[3] * mulGain[2] + invCAT2[4] * mulGain[5] + invCAT2[5] * mulGain[8],

			invCAT2[6] * mulGain[0] + invCAT2[7] * mulGain[3] + invCAT2[8] * mulGain[6],
			invCAT2[6] * mulGain[1] + invCAT2[7] * mulGain[4] + invCAT2[8] * mulGain[7],
			invCAT2[6] * mulGain[2] + invCAT2[7] * mulGain[5] + invCAT2[8] * mulGain[8],
		};

		const double mult[9] = // in future must be enclosed in C++ class Matrix
		{
			XYZtosRGB[0] * outMatrix[0] + XYZtosRGB[1] * outMatrix[3] + XYZtosRGB[2] * outMatrix[6],
			XYZtosRGB[0] * outMatrix[1] + XYZtosRGB[1] * outMatrix[4] + XYZtosRGB[2] * outMatrix[7],
			XYZtosRGB[0] * outMatrix[2] + XYZtosRGB[1] * outMatrix[5] + XYZtosRGB[2] * outMatrix[8],

			XYZtosRGB[3] * outMatrix[0] + XYZtosRGB[4] * outMatrix[3] + XYZtosRGB[5] * outMatrix[6],
			XYZtosRGB[3] * outMatrix[1] + XYZtosRGB[4] * outMatrix[4] + XYZtosRGB[5] * outMatrix[7],
			XYZtosRGB[3] * outMatrix[2] + XYZtosRGB[4] * outMatrix[5] + XYZtosRGB[5] * outMatrix[8],

			XYZtosRGB[6] * outMatrix[0] + XYZtosRGB[7] * outMatrix[3] + XYZtosRGB[8] * outMatrix[6],
			XYZtosRGB[6] * outMatrix[1] + XYZtosRGB[7] * outMatrix[4] + XYZtosRGB[8] * outMatrix[7],
			XYZtosRGB[6] * outMatrix[2] + XYZtosRGB[7] * outMatrix[5] + XYZtosRGB[8] * outMatrix[8]
		};

		const double correctionMatrix[9] = // in future must be enclosed in C++ class Matrix
		{
			mult[0] * sRGBtoXYZ[0] + mult[1] * sRGBtoXYZ[3] + mult[2] * sRGBtoXYZ[6],
			mult[0] * sRGBtoXYZ[1] + mult[1] * sRGBtoXYZ[4] + mult[2] * sRGBtoXYZ[7],
			mult[0] * sRGBtoXYZ[2] + mult[1] * sRGBtoXYZ[5] + mult[2] * sRGBtoXYZ[8],

			mult[3] * sRGBtoXYZ[0] + mult[4] * sRGBtoXYZ[3] + mult[5] * sRGBtoXYZ[6],
			mult[3] * sRGBtoXYZ[1] + mult[4] * sRGBtoXYZ[4] + mult[5] * sRGBtoXYZ[7],
			mult[3] * sRGBtoXYZ[2] + mult[4] * sRGBtoXYZ[5] + mult[5] * sRGBtoXYZ[8],

			mult[6] * sRGBtoXYZ[0] + mult[7] * sRGBtoXYZ[3] + mult[8] * sRGBtoXYZ[6],
			mult[6] * sRGBtoXYZ[1] + mult[7] * sRGBtoXYZ[4] + mult[8] * sRGBtoXYZ[7],
			mult[6] * sRGBtoXYZ[2] + mult[7] * sRGBtoXYZ[5] + mult[8] * sRGBtoXYZ[8]
		};

		// get again source image pointer
		srcImg = const_cast<csSDK_uint32*>(srcFrame);

		double R0, R1, R2;
		double G0, G1, G2;
		double B0, B1, B2;

		double outR0, outR1, outR2;
		double outG0, outG1, outG2;
		double outB0, outB1, outB2;

		const int fraction = width % 3;
		int alpha0, alpha1, alpha2;

		// second pass - apply color correction
		for (int j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
			for (int i = 0; i < width; i += 3)
			{
				alpha0 = *srcImg >> 24;
				R0 = static_cast<double>((*srcImg & 0x00FF0000) >> 16);
				G0 = static_cast<double>((*srcImg & 0x0000FF00) >> 8);
				B0 = static_cast<double> (*srcImg & 0x000000FF);
				srcImg++;

				alpha1 = *srcImg >> 24;
				R1 = static_cast<double>((*srcImg & 0x00FF0000) >> 16);
				G1 = static_cast<double>((*srcImg & 0x0000FF00) >> 8);
				B1 = static_cast<double> (*srcImg & 0x000000FF);
				srcImg++;

				alpha2 = *srcImg >> 24;
				R2 = static_cast<double>((*srcImg & 0x00FF0000) >> 16);
				G2 = static_cast<double>((*srcImg & 0x0000FF00) >> 8);
				B2 = static_cast<double> (*srcImg & 0x000000FF);
				srcImg++;

				outR0 = correctionMatrix[0] * R0 + correctionMatrix[1] * G0 + correctionMatrix[2] * B0;
				outR1 = correctionMatrix[0] * R1 + correctionMatrix[1] * G1 + correctionMatrix[2] * G1;
				outR2 = correctionMatrix[0] * R2 + correctionMatrix[1] * G2 + correctionMatrix[2] * B2;

				outG0 = correctionMatrix[3] * R0 + correctionMatrix[4] * G0 + correctionMatrix[5] * B0;
				outG1 = correctionMatrix[3] * R1 + correctionMatrix[4] * G1 + correctionMatrix[5] * B1;
				outG2 = correctionMatrix[3] * R2 + correctionMatrix[4] * G2 + correctionMatrix[5] * B2;

				outB0 = correctionMatrix[6] * R0 + correctionMatrix[7] * G0 + correctionMatrix[8] * B0;
				outB1 = correctionMatrix[6] * R1 + correctionMatrix[7] * G1 + correctionMatrix[8] * B1;
				outB2 = correctionMatrix[6] * R2 + correctionMatrix[7] * G2 + correctionMatrix[8] * B2;

				const csSDK_uint32 pix0 = alpha0 << 24							      |				
										  (CLAMP_RGB8(static_cast<int>(outR0))) << 16 |
										  (CLAMP_RGB8(static_cast<int>(outG0))) << 8  |
										  (CLAMP_RGB8(static_cast<int>(outB0)));

				const csSDK_uint32 pix1 = alpha1 << 24                                |
										  (CLAMP_RGB8(static_cast<int>(outR1))) << 16 |
										  (CLAMP_RGB8(static_cast<int>(outG1))) << 8  |
										  (CLAMP_RGB8(static_cast<int>(outB1)));

				const csSDK_uint32 pix2 = alpha2 << 24                                |
										  (CLAMP_RGB8(static_cast<int>(outR2))) << 16 |
										  (CLAMP_RGB8(static_cast<int>(outG2))) << 8  |
										  (CLAMP_RGB8(static_cast<int>(outB2)));
				*dstImg++ = pix0;
				*dstImg++ = pix1;
				*dstImg++ = pix2;

			} // for (int i = 0; i < width; i++)

			dstImg += linePitch - width + fraction;
			srcImg += linePitch - width + fraction;

		} // for (int j = 0; j < height; j++)


	} // for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)

	return true;
}


// in future split this function for more little API's
bool procesVUYA4444_8u_slice(VideoHandle theData,
							 const double* __restrict pMatrixIn,
							 const double* __restrict pMatrixOut)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	double U_avg[maxIterCount] = {};
	double V_avg[maxIterCount] = {};

	prRect box = { 0 };

	// Get the filter parameters
	const FilterParamHandle filterParamH = reinterpret_cast<FilterParamHandle>((*theData)->specsHandle);
	// get iteration count and gray threshold from sliders
	const csSDK_int32 iterCnt = (nullptr != filterParamH) ? (*filterParamH)->sliderIterCnt : 1;
	const csSDK_int32 grayThr = (nullptr != filterParamH) ? (*filterParamH)->sliderGrayThr : 30;
	const eILLIUMINATE setIlluminate = (nullptr != filterParamH) ? (*filterParamH)->illuminate : DAYLIGHT;

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;
	int totalGray;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	const csSDK_uint32* __restrict dstFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const int lastIter = iterCnt - 1;

	int Y, U, V;
	double F;
	double U_bar, V_bar;
	double U_diff, V_diff, normVal;
	const double T = get_gray_threshold(grayThr);
	constexpr double b = 0.0010; // convergence threshold
	constexpr double algEpsilon = 1.00e-06;

	for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)
	{
		csSDK_uint32* srcImg = const_cast<csSDK_uint32*>(srcFrame);
		// only last iteration going with double buffering
		csSDK_uint32* dstImg = const_cast<csSDK_uint32*>((lastIter == iter_cnt) ? dstFrame : srcFrame);

		U_bar = V_bar = 0.00;
		totalGray = 0;

		// first pass - accquire color statistics
		for (int j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
				for (int i = 0; i < width; i++)
				{
					Y = static_cast<int>((*srcImg & 0x00FF0000) >> 16);
					U = static_cast<int>((*srcImg & 0x0000FF00) >> 8) - 128;
					V = static_cast<int>( *srcImg & 0x000000FF) - 128;
					srcImg++;

					const double absSum = static_cast<double>(abs(U) + abs(V));
					F = absSum / Y;

					if (F < T) {
						totalGray++;
						U_bar += U;
						V_bar += V;
					}

				} // for (int i = 0; i < width; i++)

			srcImg += linePitch - width;

		} // for (int j = 0; j < height; j++)

		U_avg[iter_cnt] = U_bar / totalGray;
		V_avg[iter_cnt] = V_bar / totalGray;

		if (MAX(abs(U_avg[iter_cnt]), abs(V_avg[iter_cnt])) < b)
			return true; // converged

		if (iter_cnt >= 1)
		{
			U_diff = U_avg[iter_cnt] - U_avg[iter_cnt - 1];
			V_diff = V_avg[iter_cnt] - V_avg[iter_cnt - 1];

			normVal = sqrt(U_diff * U_diff + V_diff * V_diff);

			if (normVal < algEpsilon)
			{
				// U and V no longer improving, so just copy source to destination and break the loop
				copy_src2dst(srcFrame, const_cast<csSDK_uint32*>(dstFrame), height, width, rowbytes);
				return true; // U and V no longer improving
			}
		}

		// convert the average GRAY from YUV to RGB
		const double restored_R = 100.00 * pMatrixIn[0] + U_avg[iter_cnt] * pMatrixIn[1] + V_avg[iter_cnt] * pMatrixIn[2];
		const double restored_G = 100.00 * pMatrixIn[3] + U_avg[iter_cnt] * pMatrixIn[4] + V_avg[iter_cnt] * pMatrixIn[5];
		const double restored_B = 100.00 * pMatrixIn[6] + U_avg[iter_cnt] * pMatrixIn[7] + V_avg[iter_cnt] * pMatrixIn[8];

		// calculate xy chromaticity
		const double xX = restored_R * sRGBtoXYZ[0] + restored_G * sRGBtoXYZ[1] + restored_B * sRGBtoXYZ[2];
		const double xY = restored_R * sRGBtoXYZ[3] + restored_G * sRGBtoXYZ[4] + restored_B * sRGBtoXYZ[5];
		const double xZ = restored_R * sRGBtoXYZ[6] + restored_G * sRGBtoXYZ[7] + restored_B * sRGBtoXYZ[8];

		const double xXYZSum = xX + xY + xZ;
		const double xyEst[2] = { xX / xXYZSum, xY / xXYZSum };
		const double xyEstDiv = 100.00 / xyEst[1];

		// converts xyY chromaticity to CIE XYZ
		const double xyzEst[3] = {
			xyEstDiv * xyEst[0],
			100.00,
			xyEstDiv * (1.00 - xyEst[0] - xyEst[1])
		};

		// get illuminate
		const double* illuminate = GetIlluminate(setIlluminate);

		const double gainTarget[3] =
		{
			illuminate[0] * CAT2[0] + illuminate[1] * CAT2[1] + illuminate[2] * CAT2[2],
			illuminate[0] * CAT2[3] + illuminate[1] * CAT2[4] + illuminate[2] * CAT2[5],
			illuminate[0] * CAT2[6] + illuminate[1] * CAT2[7] + illuminate[2] * CAT2[8]
		};

		const double gainEstimated[3] =
		{
			xyzEst[0] * CAT2[0] + xyzEst[1] * CAT2[1] + xyzEst[2] * CAT2[2],
			xyzEst[0] * CAT2[3] + xyzEst[1] * CAT2[4] + xyzEst[2] * CAT2[5],
			xyzEst[0] * CAT2[6] + xyzEst[1] * CAT2[7] + xyzEst[2] * CAT2[8]
		};

		// gain = (xfm*xyz_target)./(xfm*xyz_est);
		const double finalGain[3] =
		{
			gainTarget[0] / gainEstimated[0],
			gainTarget[1] / gainEstimated[1],
			gainTarget[2] / gainEstimated[2]
		};

		const double diagGain[9] =
		{
			finalGain[0], 0.0, 0.0 ,
			0.0, finalGain[1], 0.0,
			0.0, 0.0, finalGain[2]
		};

		const double mulGain[9] = // in future must be enclosed in C++ class Matrix: diagGain * CAT2
		{
			diagGain[0] * CAT2[0] + diagGain[1] * CAT2[3] + diagGain[2] * CAT2[6],
			diagGain[0] * CAT2[1] + diagGain[1] * CAT2[4] + diagGain[2] * CAT2[7],
			diagGain[0] * CAT2[2] + diagGain[1] * CAT2[5] + diagGain[2] * CAT2[8],

			diagGain[3] * CAT2[0] + diagGain[4] * CAT2[3] + diagGain[5] * CAT2[6],
			diagGain[3] * CAT2[1] + diagGain[4] * CAT2[4] + diagGain[5] * CAT2[7],
			diagGain[3] * CAT2[2] + diagGain[4] * CAT2[5] + diagGain[5] * CAT2[8],

			diagGain[6] * CAT2[0] + diagGain[7] * CAT2[3] + diagGain[8] * CAT2[6],
			diagGain[6] * CAT2[1] + diagGain[7] * CAT2[4] + diagGain[8] * CAT2[7],
			diagGain[6] * CAT2[2] + diagGain[7] * CAT2[5] + diagGain[8] * CAT2[8]
		};

		const double outMatrix[9] = // in future must be enclosed in C++ class Matrix: invCAT2 * mulGain
		{
			invCAT2[0] * mulGain[0] + invCAT2[1] * mulGain[3] + invCAT2[2] * mulGain[6],
			invCAT2[0] * mulGain[1] + invCAT2[1] * mulGain[4] + invCAT2[2] * mulGain[7],
			invCAT2[0] * mulGain[2] + invCAT2[1] * mulGain[5] + invCAT2[2] * mulGain[8],

			invCAT2[3] * mulGain[0] + invCAT2[4] * mulGain[3] + invCAT2[5] * mulGain[6],
			invCAT2[3] * mulGain[1] + invCAT2[4] * mulGain[4] + invCAT2[5] * mulGain[7],
			invCAT2[3] * mulGain[2] + invCAT2[4] * mulGain[5] + invCAT2[5] * mulGain[8],

			invCAT2[6] * mulGain[0] + invCAT2[7] * mulGain[3] + invCAT2[8] * mulGain[6],
			invCAT2[6] * mulGain[1] + invCAT2[7] * mulGain[4] + invCAT2[8] * mulGain[7],
			invCAT2[6] * mulGain[2] + invCAT2[7] * mulGain[5] + invCAT2[8] * mulGain[8],
		};

		const double mult[9] = // in future must be enclosed in C++ class Matrix
		{
			XYZtosRGB[0] * outMatrix[0] + XYZtosRGB[1] * outMatrix[3] + XYZtosRGB[2] * outMatrix[6],
			XYZtosRGB[0] * outMatrix[1] + XYZtosRGB[1] * outMatrix[4] + XYZtosRGB[2] * outMatrix[7],
			XYZtosRGB[0] * outMatrix[2] + XYZtosRGB[1] * outMatrix[5] + XYZtosRGB[2] * outMatrix[8],

			XYZtosRGB[3] * outMatrix[0] + XYZtosRGB[4] * outMatrix[3] + XYZtosRGB[5] * outMatrix[6],
			XYZtosRGB[3] * outMatrix[1] + XYZtosRGB[4] * outMatrix[4] + XYZtosRGB[5] * outMatrix[7],
			XYZtosRGB[3] * outMatrix[2] + XYZtosRGB[4] * outMatrix[5] + XYZtosRGB[5] * outMatrix[8],

			XYZtosRGB[6] * outMatrix[0] + XYZtosRGB[7] * outMatrix[3] + XYZtosRGB[8] * outMatrix[6],
			XYZtosRGB[6] * outMatrix[1] + XYZtosRGB[7] * outMatrix[4] + XYZtosRGB[8] * outMatrix[7],
			XYZtosRGB[6] * outMatrix[2] + XYZtosRGB[7] * outMatrix[5] + XYZtosRGB[8] * outMatrix[8]
		};


		const double correctionMatrix[9] = // in future must be enclosed in C++ class Matrix
		{
			mult[0] * sRGBtoXYZ[0] + mult[1] * sRGBtoXYZ[3] + mult[2] * sRGBtoXYZ[6],
			mult[0] * sRGBtoXYZ[1] + mult[1] * sRGBtoXYZ[4] + mult[2] * sRGBtoXYZ[7],
			mult[0] * sRGBtoXYZ[2] + mult[1] * sRGBtoXYZ[5] + mult[2] * sRGBtoXYZ[8],

			mult[3] * sRGBtoXYZ[0] + mult[4] * sRGBtoXYZ[3] + mult[5] * sRGBtoXYZ[6],
			mult[3] * sRGBtoXYZ[1] + mult[4] * sRGBtoXYZ[4] + mult[5] * sRGBtoXYZ[7],
			mult[3] * sRGBtoXYZ[2] + mult[4] * sRGBtoXYZ[5] + mult[5] * sRGBtoXYZ[8],

			mult[6] * sRGBtoXYZ[0] + mult[7] * sRGBtoXYZ[3] + mult[8] * sRGBtoXYZ[6],
			mult[6] * sRGBtoXYZ[1] + mult[7] * sRGBtoXYZ[4] + mult[8] * sRGBtoXYZ[7],
			mult[6] * sRGBtoXYZ[2] + mult[7] * sRGBtoXYZ[5] + mult[8] * sRGBtoXYZ[8]
		};

		// get again source image pointer
		srcImg = const_cast<csSDK_uint32*>(srcFrame);

		double Y0, Y1, Y2;
		double U0, U1, U2;
		double V0, V1, V2;

		double R0, R1, R2;
		double G0, G1, G2;
		double B0, B1, B2;

		double outR0, outR1, outR2;
		double outG0, outG1, outG2;
		double outB0, outB1, outB2;

		double outY0, outY1, outY2;
		double outU0, outU1, outU2;
		double outV0, outV1, outV2;

		const double* __restrict pYUV2RGB = pMatrixIn;
		const double* __restrict pRGB2YUV = pMatrixOut;

		const int fraction = width % 3;
		int alpha0, alpha1, alpha2;

		// second pass - apply color correction
		for (int j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
				for (int i = 0; i < width; i += 3)
				{
					alpha0 = *srcImg >> 24;
					Y0 = static_cast<int>((*srcImg & 0x00FF0000) >> 16);
					U0 = static_cast<int>((*srcImg & 0x0000FF00) >> 8)  - 128;
					V0 = static_cast<int> (*srcImg & 0x000000FF) - 128;
					srcImg++;

					alpha1 = *srcImg >> 24;
					Y1 = static_cast<int>((*srcImg & 0x00FF0000) >> 16);
					U1 = static_cast<int>((*srcImg & 0x0000FF00) >> 8)  - 128;
					V1 = static_cast<int> (*srcImg & 0x000000FF) - 128;
					srcImg++;

					alpha2 = *srcImg >> 24;
					Y2 = static_cast<int>((*srcImg & 0x00FF0000) >> 16);
					U2 = static_cast<int>((*srcImg & 0x0000FF00) >> 8)  - 128;
					V2 = static_cast<int> (*srcImg & 0x000000FF) - 128;
					srcImg++;

					R0 = Y0 * pYUV2RGB[0] + U0 * pYUV2RGB[1] + V0 * pYUV2RGB[2];
					G0 = Y0 * pYUV2RGB[3] + U0 * pYUV2RGB[4] + V0 * pYUV2RGB[5];
					B0 = Y0 * pYUV2RGB[6] + U0 * pYUV2RGB[7] + V0 * pYUV2RGB[8];

					R1 = Y1 * pYUV2RGB[0] + U1 * pYUV2RGB[1] + V1 * pYUV2RGB[2];
					G1 = Y1 * pYUV2RGB[3] + U1 * pYUV2RGB[4] + V1 * pYUV2RGB[5];
					B1 = Y1 * pYUV2RGB[6] + U1 * pYUV2RGB[7] + V1 * pYUV2RGB[8];

					R2 = Y2 * pYUV2RGB[0] + U2 * pYUV2RGB[1] + V2 * pYUV2RGB[2];
					G2 = Y2 * pYUV2RGB[3] + U2 * pYUV2RGB[4] + V2 * pYUV2RGB[5];
					B2 = Y2 * pYUV2RGB[6] + U2 * pYUV2RGB[7] + V2 * pYUV2RGB[8];

					outR0 = correctionMatrix[0] * R0 + correctionMatrix[1] * G0 + correctionMatrix[2] * B0;
					outG0 = correctionMatrix[3] * R0 + correctionMatrix[4] * G0 + correctionMatrix[5] * B0;
					outB0 = correctionMatrix[6] * R0 + correctionMatrix[7] * G0 + correctionMatrix[8] * B0;

					outR1 = correctionMatrix[0] * R1 + correctionMatrix[1] * G1 + correctionMatrix[2] * G1;
					outG1 = correctionMatrix[3] * R1 + correctionMatrix[4] * G1 + correctionMatrix[5] * B1;
					outB1 = correctionMatrix[6] * R1 + correctionMatrix[7] * G1 + correctionMatrix[8] * B1;

					outR2 = correctionMatrix[0] * R2 + correctionMatrix[1] * G2 + correctionMatrix[2] * B2;
					outG2 = correctionMatrix[3] * R2 + correctionMatrix[4] * G2 + correctionMatrix[5] * B2;
					outB2 = correctionMatrix[6] * R2 + correctionMatrix[7] * G2 + correctionMatrix[8] * B2;

					outY0 = outR0 * pRGB2YUV[0] + outG0 * pRGB2YUV[1] + outB0 * pRGB2YUV[2];
					outU0 = outR0 * pRGB2YUV[3] + outG0 * pRGB2YUV[4] + outB0 * pRGB2YUV[5] + 128;
					outV0 = outR0 * pRGB2YUV[6] + outG0 * pRGB2YUV[7] + outB0 * pRGB2YUV[8] + 128;

					outY1 = outR1 * pRGB2YUV[0] + outG1 * pRGB2YUV[1] + outB1 * pRGB2YUV[2];
					outU1 = outR1 * pRGB2YUV[3] + outG1 * pRGB2YUV[4] + outB1 * pRGB2YUV[5] + 128;
					outV1 = outR1 * pRGB2YUV[6] + outG1 * pRGB2YUV[7] + outB1 * pRGB2YUV[8] + 128;

					outY2 = outR2 * pRGB2YUV[0] + outG2 * pRGB2YUV[1] + outB2 * pRGB2YUV[2];
					outU2 = outR2 * pRGB2YUV[3] + outG2 * pRGB2YUV[4] + outB2 * pRGB2YUV[5] + 128;
					outV2 = outR2 * pRGB2YUV[6] + outG2 * pRGB2YUV[7] + outB2 * pRGB2YUV[8] + 128;
					

					const csSDK_uint32 pix0 = alpha0	    		  << 24 |
								(CLAMP_RGB8(static_cast<int>(outY0))) << 16 |
								(CLAMP_RGB8(static_cast<int>(outU0))) <<  8 |
								(CLAMP_RGB8(static_cast<int>(outV0)));

					const csSDK_uint32 pix1 = alpha1				  << 24 |
								(CLAMP_RGB8(static_cast<int>(outY1))) << 16 |
								(CLAMP_RGB8(static_cast<int>(outU1))) <<  8 |
								(CLAMP_RGB8(static_cast<int>(outV1)));

					const csSDK_uint32 pix2 = alpha2                  << 24 |
								(CLAMP_RGB8(static_cast<int>(outY2))) << 16 |
								(CLAMP_RGB8(static_cast<int>(outU2))) << 8  |
								(CLAMP_RGB8(static_cast<int>(outV2)));

					*dstImg++ = pix0;
					*dstImg++ = pix1;
					*dstImg++ = pix2;

				} // for (int i = 0; i < width; i++)

			dstImg += linePitch - width + fraction;
			srcImg += linePitch - width + fraction;

		} // for (int j = 0; j < height; j++)

	} // for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)

	return true;
}