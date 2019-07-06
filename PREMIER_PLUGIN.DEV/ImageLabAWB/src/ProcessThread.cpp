#include "AdobeImageLabAWB.h"

// define color space conversion matrix's
CACHE_ALIGN double constexpr RGB2YUV[LAST][9] =
{
	// BT.601
	{ 
	  0.299000,  0.587000,  0.114000,
	 -0.147130, -0.288860,  0.436000,
	  0.615000, -0.514990, -0.100010
	},

	// BT.709
	{ 
	  0.212600,   0.715200,  0.072200,
	 -0.099910,  -0.336090,  0.436000,
	  0.615000,  -0.558610, -0.056390  
	},

	// BT.2020
	{
  	  0.262700,   0.678000,  0.059300,
     -0.139600,  -0.360400,  0.500000,
	  0.500000,  -0.459800, -0.040200 
	},

	// SMPTE 240M
	{
	  0.212200,   0.701300,  0.086500,
     -0.116200,  -0.383800,  0.500000,
	  0.500000,  -0.445100, -0.054900 
	}
};



CACHE_ALIGN double constexpr YUV2RGB[LAST][9] =
{


};


bool procesBGRA4444_8u_slice(VideoHandle theData)
{
	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const double* __restrict const coefficientsBT601 = RGB2YUV[STD_BT601];

	CACHE_ALIGN byte yuv601Buffer[3][width];

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			const csSDK_uint32 BGRAPixel = *srcPix++;
			const byte r =  BGRAPixel & 0x000000FFu;
			const byte g = (BGRAPixel & 0x0000FF00u) >> 8;
			const byte b = (BGRAPixel & 0x00FF0000u) >> 16;

			const double R = static_cast<double>(r);
			const double G = static_cast<double>(g);
			const double B = static_cast<double>(b);

			const double Y = R * coefficientsBT601[0] +
							 G * coefficientsBT601[1] +
							 B * coefficientsBT601[2];

			const double U = R * coefficientsBT601[3] +
							 G * coefficientsBT601[4] +
							 B * coefficientsBT601[5];

			const double V = R * coefficientsBT601[6] +
							 G * coefficientsBT601[7] +
				             B * coefficientsBT601[8];
		}
	}

	return true;
}