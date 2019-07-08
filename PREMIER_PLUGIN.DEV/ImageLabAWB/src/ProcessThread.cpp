#include "AdobeImageLabAWB.h"




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
	double F;
	double U_diff, V_diff, normVal;
	double U_bar, V_bar;
	constexpr double T = 0.30; // will be slider position
	constexpr double b = 0.0010; // convergence threshold
	constexpr double algEpsilon = 1.0000e-06;

	for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)
	{
		csSDK_uint32* srcImg = const_cast<csSDK_uint32*>(srcFrame);
		// only last iteration going with double buffering, before last - in-place operations
		csSDK_uint32* dstImg = const_cast<csSDK_uint32*>((iter_cnt == lastIter) ? dstFrame : srcFrame);

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

	} // for (int iter_cnt = 0; iter_cnt < iterCnt; iter_cnt++)

	return true;
}