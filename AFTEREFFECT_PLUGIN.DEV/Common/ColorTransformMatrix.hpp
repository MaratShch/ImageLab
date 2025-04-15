#pragma once

#include "Common.hpp"

typedef enum eCOLOR_SPACE
{
	BT601 = 0,
	BT709,
	BT2020,
	SMPTE240M
};

typedef enum eCOLOR_OBSERVER
{
	observer_CIE_1931 = 0, /*  2° (CIE 1931) */
	observer_CIE_1964,     /* 10° (CIE 1964) */
	observer_TOTAL_OBSERVERS
};


typedef enum eCOLOR_ILLUMINANT
{
	color_ILLUMINANT_A = 0,
	color_ILLUMINANT_B,
	color_ILLUMINANT_C,
	color_ILLUMINANT_D50,
	color_ILLUMINANT_D55,
	color_ILLUMINANT_D65,
	color_ILLUMINANT_D75,
	color_ILLUMINANT_E,
	color_ILLUMINANT_F1,
	color_ILLUMINANT_F2,
	color_ILLUMINANT_F3,
	color_ILLUMINANT_F4,
	color_ILLUMINANT_F5,
	color_ILLUMINANT_F6,
	color_ILLUMINANT_F7,
	color_ILLUMINANT_F8,
	color_ILLUMINANT_F9,
	color_ILLUMINANT_F10,
	color_ILLUMINANT_F11,
	color_ILLUMINANT_F12,
	color_TOTAL_ILLUMINANTS
};

constexpr eCOLOR_OBSERVER   CieLabDefaultObserver  { observer_CIE_1931 };
constexpr eCOLOR_ILLUMINANT CieLabDefaultIlluminant{ color_ILLUMINANT_D65 };

// define color space conversion matrix's
CACHE_ALIGN float constexpr RGB2YUV[][9] =
{
	// BT.601
	{
		0.299000f,  0.587000f,  0.114000f,
	   -0.168736f, -0.331264f,  0.500000f,
		0.500000f, -0.418688f, -0.081312f
	},

	// BT.709
	{
		0.212600f,   0.715200f,  0.072200f,
	   -0.114570f,  -0.385430f,  0.500000f,
		0.500000f,  -0.454150f, -0.045850f
	},

	// BT.2020
	{
		0.262700f,   0.678000f,  0.059300f,
	   -0.139630f,  -0.360370f,  0.500000f,
		0.500000f,  -0.459790f, -0.040210f
	},

	// SMPTE 240M
	{
		0.212200f,   0.701300f,  0.086500f,
	   -0.116200f,  -0.383800f,  0.500000f,
		0.500000f,  -0.445100f, -0.054900f
	}
};

CACHE_ALIGN float constexpr YUV2RGB[][9] =
{
	// BT.601
	{
		1.000000f,  0.000000f,  1.407500f,
		1.000000f, -0.344140f, -0.716900f,
		1.000000f,  1.779000f,  0.000000f
	},

	// BT.709
	{
		1.000000f,  0.00000000f,  1.5748021f,
		1.000000f, -0.18732698f, -0.4681240f,
		1.000000f,  1.85559927f,  0.0000000f
	},

	// BT.2020
	{
		1.000000f,  0.00000000f,  1.4745964f,
		1.000000f, -0.16454810f, -0.5713517f,
		1.000000f,  1.88139998f,  0.0000000f
	},

	// SMPTE 240M
	{
		1.000000f,  0.0000000f,  1.5756000f,
		1.000000f, -0.2253495f, -0.4767712f,
		1.000000f,  1.8270219f,  0.0000000f
	}
};


CACHE_ALIGN constexpr float sRGBtoXYZ[9] = 
{
    0.4124564f,  0.3575761f,  0.1804375f,
    0.2126729f,  0.7151522f,  0.0721750f,
    0.0193339f,  0.1191920f,  0.9503041f
};

CACHE_ALIGN constexpr float XYZtosRGB[9] = 
{
    3.240455f, -1.537139f, -0.498532f,
   -0.969266f,  1.876011f,  0.041556f,
	0.055643f, -0.204026f,  1.057225f
};

CACHE_ALIGN constexpr float yuv2xyz[3] = { 0.114653800f, 0.083911980f, 0.082220770f };
CACHE_ALIGN constexpr float xyz2yuv[3] = { 0.083911980f, 0.283096500f, 0.466178900f };



CACHE_ALIGN constexpr float cCOLOR_ILLUMINANT[observer_TOTAL_OBSERVERS][color_TOTAL_ILLUMINANTS][3] =
{
	/* 2° (CIE 1931) */
	{
		/* A */
		{ 109.850f, 100.000f, 35.585f },
		/* B */
		{ 99.0927f,	100.000f, 85.313f },
		/* C */
		{ 98.074f,	100.000f, 118.232f },
		/* D50 */
		{ 96.422f,	100.000f, 82.521f },
		/* D55 */
		{ 95.682f,  100.000f, 92.149f },
		/* D65 */
		{ 95.047f,	100.000f, 108.883f },
		/* D75 */
		{ 94.972f,  100.000f, 122.638f },
		/* E */
		{ 100.000f,	100.000f, 100.000f },
		/* F1 */
		{ 92.834f,	100.000f, 103.665f },
		/* F2 */
		{ 99.187f,	100.000f, 67.395f },
		/* F3 */
		{ 103.754f,	100.000f, 49.861f },
		/* F4 */
		{ 109.147f,	100.000f, 38.813f },
		/* F5 */
		{ 90.872f,	100.000f, 98.723f },
		/* F6 */
		{ 97.309f,	100.000f, 60.191f },
		/* F7 */
		{ 95.044f,	100.000f, 108.755f },
		/* F8 */
		{ 96.413f,	100.000f, 82.333f },
		/* F9 */
		{ 100.365f,	100.000f, 67.868f },
		/* F10 */
		{ 96.174f,	100.000f, 81.712f },
		/* F11 */
		{ 100.966f,	100.000f, 64.370f },
		/* F12 */
		{ 108.046f, 100.000f, 39.228f }
    },

	/* 10° (CIE 1964) */
	{
		/* A */
		{ 111.144f,	100.000f, 35.200f },
		/* B */
		{ 99.178f,	100.000f, 84.3493f },
		/* C */
		{ 97.285f,	100.000f, 116.145f },
		/* D50 */
		{ 96.720f,	100.000f, 81.427f },
		/* D55 */
		{ 95.799f,	100.000f, 90.926f },
		/* D65 */
		{ 94.811f,	100.000f, 107.304 },
		/* D75 */
		{ 94.416f,	100.000f, 120.641 },
		/* E */
		{ 100.000f,	100.000f, 100.000f },
		/* F1 */
		{ 94.791f,	100.000f, 103.191f },
		/* F2 */
		{ 103.280f,	100.000f, 69.026f },
		/* F3 */
		{ 108.968f,	100.000f, 51.965f },
		/* F4 */
		{ 114.961f,	100.000f, 40.963f },
		/* F5 */
		{ 93.369f,	100.000f, 98.636f },
		/* F6 */
		{ 102.148f,	100.000f, 62.074f },
		/* F7 */
		{ 95.792f,	100.000f, 107.687f },
		/* F8 */
		{ 97.115f,	100.000f, 81.135f },
		/* F9 */
		{ 102.116f,	100.000f, 67.826f },
		/* F10 */
		{ 99.001f,	100.000f, 83.134f },
		/* F11 */
		{ 103.866f,	100.000f, 65.627f },
		/* F12 */
		{ 111.428f,	100.000f, 40.353f }
    }
};
