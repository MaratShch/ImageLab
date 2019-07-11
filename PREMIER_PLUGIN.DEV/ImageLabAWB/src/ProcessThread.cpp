#include "AdobeImageLabAWB.h"


CACHE_ALIGN constexpr double sRGBtoXYZ[9] = {
	0.4124564,  0.3575761,  0.1804375,
	0.2126729,  0.7151522,  0.0721750,
	0.0193339,  0.1191920,  0.9503041
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


bool procesBGRA4444_8u_slice(	VideoHandle theData, 
								const double* __restrict pMatrixIn,
								const double* __restrict pMatrixOut,
								const int                iterCnt)
{
	CACHE_ALIGN double U_avg[maxIterCount];
	CACHE_ALIGN double V_avg[maxIterCount];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	const csSDK_uint32* __restrict dstFrame = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	int totalGray = 0;
	const int lastIter = iterCnt - 1;

	double R, G, B;
	double Y, U, V;
	double xX, xY, xZ;
	double F;
	double U_diff, V_diff, normVal;
	double U_bar, V_bar;
	constexpr double T = 0.30; // will be slider position
	constexpr double b = 0.0010; // convergence threshold
	constexpr double algEpsilon = 1.00e-06;

	for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)
	{
		csSDK_uint32* srcImg = const_cast<csSDK_uint32*>(srcFrame);
		// only last iteration going with double buffering, before last - in-place operations
		csSDK_uint32* dstImg = const_cast<csSDK_uint32*>((lastIter == iter_cnt) ? dstFrame : srcFrame);

		U_bar = V_bar = 0.00;

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
			break; // converged

		if (iter_cnt >= 1)
		{
			U_diff = U_avg[iter_cnt] - U_avg[iter_cnt - 1];
			V_diff = V_avg[iter_cnt] - V_avg[iter_cnt - 1];

//			normVal = asqrt(U_diff * U_diff + V_diff * V_diff);
			normVal = sqrt(U_diff * U_diff + V_diff * V_diff);

			if (normVal < algEpsilon)
				break; // U and V no longer improving
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
		const double* illuminate = GetIlluminate();

		double gainTarget[9];
		double gainEstimated[9];

		gainTarget[0] = illuminate[0] * CAT2[0] + illuminate[1] * CAT2[1] + illuminate[2] * CAT2[2];
		gainTarget[1] = illuminate[0] * CAT2[3] + illuminate[1] * CAT2[4] + illuminate[2] * CAT2[5];
		gainTarget[2] = illuminate[0] * CAT2[6] + illuminate[1] * CAT2[7] + illuminate[2] * CAT2[8];

		gainEstimated[0] = xyzEst[0] * CAT2[0] + xyzEst[1] * CAT2[1] + xyzEst[2] * CAT2[2];
		gainEstimated[1] = xyzEst[0] * CAT2[3] + xyzEst[1] * CAT2[4] + xyzEst[2] * CAT2[5];
		gainEstimated[2] = xyzEst[0] * CAT2[6] + xyzEst[1] * CAT2[7] + xyzEst[2] * CAT2[8];

		const double finalGain[3] = 
		{
			gainTarget[0] / gainEstimated[0],
			gainTarget[1] / gainEstimated[1],
			gainTarget[2] / gainEstimated[2]
		};



	} // for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)

	return true;
}