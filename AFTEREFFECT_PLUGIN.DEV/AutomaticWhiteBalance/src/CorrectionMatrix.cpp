#include "AutomaticWhiteBalance.hpp"


const float* __restrict const GetIlluminate(const eILLUMINATE& illuminateIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblIlluminate[12][3] = {
		{ 0.f },                             // NONE    
		{ 95.0470f,  100.0000f, 108.8830f }, // DAYLIGHT - D65 (DEFAULT)
		{ 98.0740f,  100.0000f, 118.2320f }, // OLD_DAYLIGHT
		{ 99.0927f,  100.0000f,  85.3130f }, // OLD_DIRECT_SUNLIGHT_AT_NOON
		{ 95.6820f,  100.0000f,  92.1490f }, // MID_MORNING_DAYLIGHT
		{ 94.9720f,  100.0000f, 122.6380f }, // NORTH_SKY_DAYLIGHT
		{ 92.8340f,  100.0000f, 103.6650f }, // DAYLIGHT_FLUORESCENT_F1
		{ 99.1870f,  100.0000f,  67.3950f }, // COOL_FLUERESCENT
		{ 103.7540f, 100.0000f,  49.8610f }, // WHITE_FLUORESCENT
		{ 109.1470f, 100.0000f,  38.8130f }, // WARM_WHITE_FLUORESCENT
		{ 90.8720f,  100.0000f,  98.7230f }, // DAYLIGHT_FLUORESCENT_F5
		{ 100.3650f, 100.0000f,  67.8680f }  // COOL_WHITE_FLUORESCENT
	};

	return tblIlluminate[illuminateIdx];
}

const float* __restrict GetColorAdaptation(const eChromaticAdaptation& adaptationIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblColorAdaptation[5][9] = {
		{ 0.73280f,  0.4296f, -0.16240f, -0.7036f, 1.69750f, 0.0061f, 0.0030f,  0.0136f, 0.98340f }, // CAT-02
		{ 0.40024f,  0.7076f, -0.08081f, -0.2263f, 1.16532f, 0.0457f, 0.0f,     0.0f,    0.91822f }, // VON-KRIES
		{ 0.89510f,  0.2664f, -0.16140f, -0.7502f, 1.71350f, 0.0367f, 0.0389f, -0.0685f, 1.02960f }, // BRADFORD
		{ 1.26940f, -0.0988f, -0.17060f, -0.8364f, 1.80060f, 0.0357f, 0.0297f, -0.0315f, 1.00180f }, // SHARP
		{ 0.79820f,  0.3389f, -0.13710f, -0.5918f, 1.55120f, 0.0406f, 0.0008f,  0.2390f, 0.97530f }, // CMCCAT2000
	};

	return tblColorAdaptation[adaptationIdx];
}


const float* __restrict GetColorAdaptationInv(const eChromaticAdaptation& invAdaptationIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblColorAdaptationInv[5][9] = {
		{ 1.096124f, -0.278869f, 0.182745f,	0.454369f, 0.473533f,  0.072098f, -0.009628f, -0.005698f, 1.015326f }, // INV CAT-02
		{ 1.859936f, -1.129382f, 0.219897f, 0.361191f, 0.638812f,  0.0f,       0.0f,       0.0f,      1.089064f }, // INV VON-KRIES
		{ 0.986993f, -0.147054f, 0.159963f, 0.432305f, 0.518360f,  0.049291f, -0.008529f,  0.040043f, 0.968487f }, // INV BRADFORD
		{ 0.815633f,  0.047155f, 0.137217f, 0.379114f, 0.576942f,  0.044001f, -0.012260f,  0.016743f, 0.995519f }, // INV SHARP
		{ 1.062305f, -0.256743f, 0.160018f, 0.407920f, 0.55023f,   0.034437f, -0.100833f, -0.134626f, 1.016755f }, // INV CMCCAT2000
	};
	return tblColorAdaptationInv[invAdaptationIdx];
}


void compute_correction_matrix
(
	const float& uAvg,
	const float& vAvg,
	const eCOLOR_SPACE& colorSpaceIdx,
	const eILLUMINATE&  illuminateIdx,
	const eChromaticAdaptation& chromaticIdx,
	float* __restrict correctionMatrix /* pointer for hold correction matrix (3 values as minimal) */
) noexcept
{
	const float* __restrict colorMatrixOut = YUV2RGB[colorSpaceIdx];
    const float* __restrict illuminate = GetIlluminate(illuminateIdx);
    const float* __restrict colorAdaptation = GetColorAdaptation(chromaticIdx);
    const float* __restrict colorAdaptationInv = GetColorAdaptationInv(chromaticIdx);

	const float restored_R = 100.0f * colorMatrixOut[0] + uAvg * colorMatrixOut[1] + vAvg * colorMatrixOut[2];
	const float restored_G = 100.0f * colorMatrixOut[3] + uAvg * colorMatrixOut[4] + vAvg * colorMatrixOut[5];
	const float restored_B = 100.0f * colorMatrixOut[6] + uAvg * colorMatrixOut[7] + vAvg * colorMatrixOut[8];

	// calculate xy chromaticity
	const float xX = restored_R * sRGBtoXYZ[0] + restored_G * sRGBtoXYZ[1] + restored_B * sRGBtoXYZ[2];
	const float xY = restored_R * sRGBtoXYZ[3] + restored_G * sRGBtoXYZ[4] + restored_B * sRGBtoXYZ[5];
	const float xZ = restored_R * sRGBtoXYZ[6] + restored_G * sRGBtoXYZ[7] + restored_B * sRGBtoXYZ[8];

	const float xXYZSum = xX + xY + xZ;
	const float xyEst[2] = { xX / xXYZSum, xY / xXYZSum };
	const float xyEstDiv = 100.0f / xyEst[1];

	// Converts xyY chromaticity to CIE XYZ.
	const float xyzEst[3] = { xyEstDiv * xyEst[0],
		100.0f,
		xyEstDiv * (1.0f - xyEst[0] - xyEst[1]) };

	const float gainTarget[3] =
	{
		illuminate[0] * colorAdaptation[0] + illuminate[1] * colorAdaptation[1] + illuminate[2] * colorAdaptation[2],
		illuminate[0] * colorAdaptation[3] + illuminate[1] * colorAdaptation[4] + illuminate[2] * colorAdaptation[5],
		illuminate[0] * colorAdaptation[6] + illuminate[1] * colorAdaptation[7] + illuminate[2] * colorAdaptation[8]
	};

	const float gainEstimated[3] =
	{
		xyzEst[0] * colorAdaptation[0] + xyzEst[1] * colorAdaptation[1] + xyzEst[2] * colorAdaptation[2],
		xyzEst[0] * colorAdaptation[3] + xyzEst[1] * colorAdaptation[4] + xyzEst[2] * colorAdaptation[5],
		xyzEst[0] * colorAdaptation[6] + xyzEst[1] * colorAdaptation[7] + xyzEst[2] * colorAdaptation[8]
	};

	const float finalGain[3] =
	{
		gainTarget[0] / gainEstimated[0],
		gainTarget[1] / gainEstimated[1],
		gainTarget[2] / gainEstimated[2]
	};

	const float diagGain[9] =
	{
		finalGain[0], 0.0f, 0.0f,
		0.0f, finalGain[1], 0.0f,
		0.0f, 0.0f, finalGain[2]
	};

	const float mulGain[9] =
	{
		diagGain[0] * colorAdaptation[0] + diagGain[1] * colorAdaptation[3] + diagGain[2] * colorAdaptation[6],
		diagGain[0] * colorAdaptation[1] + diagGain[1] * colorAdaptation[4] + diagGain[2] * colorAdaptation[7],
		diagGain[0] * colorAdaptation[2] + diagGain[1] * colorAdaptation[5] + diagGain[2] * colorAdaptation[8],

		diagGain[3] * colorAdaptation[0] + diagGain[4] * colorAdaptation[3] + diagGain[5] * colorAdaptation[6],
		diagGain[3] * colorAdaptation[1] + diagGain[4] * colorAdaptation[4] + diagGain[5] * colorAdaptation[7],
		diagGain[3] * colorAdaptation[2] + diagGain[4] * colorAdaptation[5] + diagGain[5] * colorAdaptation[8],

		diagGain[6] * colorAdaptation[0] + diagGain[7] * colorAdaptation[3] + diagGain[8] * colorAdaptation[6],
		diagGain[6] * colorAdaptation[1] + diagGain[7] * colorAdaptation[4] + diagGain[8] * colorAdaptation[7],
		diagGain[6] * colorAdaptation[2] + diagGain[7] * colorAdaptation[5] + diagGain[8] * colorAdaptation[8]
	};

	const float outMatrix[9] =
	{
		colorAdaptationInv[0] * mulGain[0] + colorAdaptationInv[1] * mulGain[3] + colorAdaptationInv[2] * mulGain[6],
		colorAdaptationInv[0] * mulGain[1] + colorAdaptationInv[1] * mulGain[4] + colorAdaptationInv[2] * mulGain[7],
		colorAdaptationInv[0] * mulGain[2] + colorAdaptationInv[1] * mulGain[5] + colorAdaptationInv[2] * mulGain[8],

		colorAdaptationInv[3] * mulGain[0] + colorAdaptationInv[4] * mulGain[3] + colorAdaptationInv[5] * mulGain[6],
		colorAdaptationInv[3] * mulGain[1] + colorAdaptationInv[4] * mulGain[4] + colorAdaptationInv[5] * mulGain[7],
		colorAdaptationInv[3] * mulGain[2] + colorAdaptationInv[4] * mulGain[5] + colorAdaptationInv[5] * mulGain[8],

		colorAdaptationInv[6] * mulGain[0] + colorAdaptationInv[7] * mulGain[3] + colorAdaptationInv[8] * mulGain[6],
		colorAdaptationInv[6] * mulGain[1] + colorAdaptationInv[7] * mulGain[4] + colorAdaptationInv[8] * mulGain[7],
		colorAdaptationInv[6] * mulGain[2] + colorAdaptationInv[7] * mulGain[5] + colorAdaptationInv[8] * mulGain[8],
	};

	const float mult[9] =
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

	if (nullptr != correctionMatrix)
	{
		correctionMatrix[0] = mult[0] * sRGBtoXYZ[0] + mult[1] * sRGBtoXYZ[3] + mult[2] * sRGBtoXYZ[6];
		correctionMatrix[1] = mult[3] * sRGBtoXYZ[1] + mult[4] * sRGBtoXYZ[4] + mult[5] * sRGBtoXYZ[7];
		correctionMatrix[2] = mult[6] * sRGBtoXYZ[2] + mult[7] * sRGBtoXYZ[5] + mult[8] * sRGBtoXYZ[8];
	}

	return;
}
