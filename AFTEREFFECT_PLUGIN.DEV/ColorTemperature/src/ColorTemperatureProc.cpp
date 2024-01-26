#include "Common.hpp"
#include "ColorTenmperatureProc.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"

constexpr auto size_array = 31;

template <typename T>
struct UVT
{
	T  u;
	T  v;
	T  t;
};

using fUVT  = UVT<float>;
using dUVT  = UVT<double>;
using ldUVT = UVT<long double>;


constexpr CACHE_ALIGN dUVT uvt_f64[size_array] =
{
	{ 0.18006, 0.26352, -0.24341 },
	{ 0.18066, 0.26589, -0.25479 },
	{ 0.18133, 0.26846, -0.26876 },
	{ 0.18208, 0.27119, -0.28539 },
	{ 0.18293, 0.27407, -0.30470 },
	{ 0.18388, 0.27709, -0.32675 },
	{ 0.18494, 0.28021, -0.35156 },
	{ 0.18611, 0.28342, -0.37915 },
	{ 0.18740, 0.28668, -0.40955 },
	{ 0.18880, 0.28997, -0.44278 },
	{ 0.19032, 0.29326, -0.47888 },
	{ 0.19462, 0.30141, -0.58204 },
	{ 0.19962, 0.30921, -0.70471 },
	{ 0.20525, 0.31647, -0.84901 },
	{ 0.21142, 0.32312, -1.01820 },
	{ 0.21807, 0.32909, -1.21680 },
	{ 0.22511, 0.33439, -1.45120 },
	{ 0.23247, 0.33904, -1.72980 },
	{ 0.24010, 0.34308, -2.06370 },
	{ 0.24792, 0.34655, -2.46810 },
	{ 0.25591, 0.34951, -2.96410 },
	{ 0.26400, 0.35200, -3.58140 },
	{ 0.27218, 0.35407, -4.36330 },
	{ 0.28039, 0.35577, -5.37620 },
	{ 0.28863, 0.35714, -6.72620 },
	{ 0.29685, 0.35823, -8.59550 },
	{ 0.30505, 0.35907, -11.3240 },
	{ 0.31320, 0.35968, -15.6280 },
	{ 0.32129, 0.36011, -23.3250 },
	{ 0.32931, 0.36038, -40.7700 },
	{ 0.33724, 0.36051, -116.450 }
};

constexpr CACHE_ALIGN double rt_f64[size_array] =
{
	DBL_MIN,
	10.0e-6f,  20.0e-6f,  30.0e-6f,
	40.0e-6f,  50.0e-6f,  60.0e-6f,
	70.0e-6f,  80.0e-6f,  90.0e-6f,
	100.0e-6f, 125.0e-6f, 150.0e-6f,
	175.0e-6f, 200.0e-6f, 225.0e-6f,
	250.0e-6f, 275.0e-6f, 300.0e-6f,
	325.0e-6f, 350.0e-6f, 375.0e-6f,
	400.0e-6f, 425.0e-6f, 450.0e-6f,
	475.0e-6f, 500.0e-6f, 525.0e-6f,
	550.0e-6f, 575.0e-6f, 600.0e-6f
};


constexpr CACHE_ALIGN fUVT uvt_f32[size_array] =
{
	{ 0.18006f, 0.26352f, -0.24341f },
	{ 0.18066f, 0.26589f, -0.25479f },
	{ 0.18133f, 0.26846f, -0.26876f },
	{ 0.18208f, 0.27119f, -0.28539f },
	{ 0.18293f, 0.27407f, -0.30470f },
	{ 0.18388f, 0.27709f, -0.32675f },
	{ 0.18494f, 0.28021f, -0.35156f },
	{ 0.18611f, 0.28342f, -0.37915f },
	{ 0.18740f, 0.28668f, -0.40955f },
	{ 0.18880f, 0.28997f, -0.44278f },
	{ 0.19032f, 0.29326f, -0.47888f },
	{ 0.19462f, 0.30141f, -0.58204f },
	{ 0.19962f, 0.30921f, -0.70471f },
	{ 0.20525f, 0.31647f, -0.84901f },
	{ 0.21142f, 0.32312f, -1.01820f },
	{ 0.21807f, 0.32909f, -1.21680f },
	{ 0.22511f, 0.33439f, -1.45120f },
	{ 0.23247f, 0.33904f, -1.72980f },
	{ 0.24010f, 0.34308f, -2.06370f },
	{ 0.24792f, 0.34655f, -2.46810f },
	{ 0.25591f, 0.34951f, -2.96410f },
	{ 0.26400f, 0.35200f, -3.58140f },
	{ 0.27218f, 0.35407f, -4.36330f },
	{ 0.28039f, 0.35577f, -5.37620f },
	{ 0.28863f, 0.35714f, -6.72620f },
	{ 0.29685f, 0.35823f, -8.59550f },
	{ 0.30505f, 0.35907f, -11.3240f },
	{ 0.31320f, 0.35968f, -15.6280f },
	{ 0.32129f, 0.36011f, -23.3250f },
	{ 0.32931f, 0.36038f, -40.7700f },
	{ 0.33724f, 0.36051f, -116.450f }
};

constexpr CACHE_ALIGN float rt_f32[size_array] =
{       
	FLT_MIN,
	10.0e-6f,  20.0e-6f,  30.0e-6f,
	40.0e-6f,  50.0e-6f,  60.0e-6f,
	70.0e-6f,  80.0e-6f,  90.0e-6f,
	100.0e-6f, 125.0e-6f, 150.0e-6f,
	175.0e-6f, 200.0e-6f, 225.0e-6f,
	250.0e-6f, 275.0e-6f, 300.0e-6f,
	325.0e-6f, 350.0e-6f, 375.0e-6f,
	400.0e-6f, 425.0e-6f, 450.0e-6f,
	475.0e-6f, 500.0e-6f, 525.0e-6f,
	550.0e-6f, 575.0e-6f, 600.0e-6f
};


/* compute with single precision */
bool XYZ_to_ColorTemperature (const fXYZPix& xyz, float* temperature) noexcept
{
	constexpr float underflowVal = 1.0e-20f;
	float p, di, dm;
	int32_t i;

	if ((xyz.X < underflowVal) && (xyz.Y < underflowVal) && (xyz.Z < underflowVal))
		return false;

	const float divider = xyz.X + 15.f * xyz.Y + 3.f * xyz.Z;
	const float us = (4.f * xyz.X) / divider;
	const float vs = (6.f * xyz.Y) / divider;

	dm = 0.f;
	for (i = 0; i < 31; i++)
	{
		const fUVT& tblEntry = uvt_f32[i];
		di = (vs - tblEntry.v) - tblEntry.t * (us - tblEntry.u);
		if ((i > 0) && (((di < 0.f) && (dm >= 0.f)) || ((di >= 0.f) && (dm < 0.f))))
			break;  /* found lines bounding (us, vs) : i-1 and i */
		dm = di;
	}

	if (i == 31)
		return false;

	auto Interpolation = [](float rt1, float rt2, float p) -> float {return (rt2 - rt1) * (p + rt1);};

	const float tPrev = uvt_f32[i - 1].t;
	const float tCurr = uvt_f32[i    ].t;

	di = di / FastCompute::Sqrt(1.f + tPrev * tPrev);
	dm = dm / FastCompute::Sqrt(1.f + tCurr * tCurr);
	p = dm / (dm - di);
	*temperature = 1.f / (Interpolation(rt_f32[i - 1], rt_f32[i], p));

	return true; 
}


/* compute with double precision */
bool XYZ_to_ColorTemperature(const dXYZPix& xyz, double* temperature) noexcept
{
	constexpr double underflowVal = 1.0e-20;
	double p, di, dm;
	int32_t i;

	if ((xyz.X < underflowVal) && (xyz.Y < underflowVal) && (xyz.Z < underflowVal))
		return false;

	const double divider = xyz.X + 15.0 * xyz.Y + 3.0 * xyz.Z;
	const double us = (4.0 * xyz.X) / divider;
	const double vs = (6.0 * xyz.Y) / divider;

	dm = 0.0;
	for (i = 0; i < 31; i++)
	{
		const dUVT& tblEntry = uvt_f64[i];
		di = (vs - tblEntry.v) - tblEntry.t * (us - tblEntry.u);
		if ((i > 0) && (((di < 0.0) && (dm >= 0.0)) || ((di >= 0.0) && (dm < 0.0))))
			break;  /* found lines bounding (us, vs) : i-1 and i */
		dm = di;
	}

	if (i == 31)
		return false;

	auto Interpolation = [](double rt1, double rt2, double p) -> double {return (rt2 - rt1) * (p + rt1); };

	const double tPrev = uvt_f64[i - 1].t;
	const double tCurr = uvt_f64[i].t;

	di = di / FastCompute::Sqrt(1.0 + tPrev * tPrev);
	dm = dm / FastCompute::Sqrt(1.0 + tCurr * tCurr);
	p = dm / (dm - di);
	*temperature = 1.0 / (Interpolation(rt_f64[i - 1], rt_f64[i], p));

	return true;
}

