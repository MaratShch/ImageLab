#include "Common.hpp"
#include <math.h> 

constexpr int tblSize8 = 256;
CACHE_ALIGN static float pTable8 [tblSize8];

void CreateColorConvertTable (void) noexcept
{
	constexpr float maxVal = static_cast<float>(tblSize8) - 1.0f;
	// create table of coefficients for rapid convert from RGB to CIELab color space
	__VECTOR_ALIGNED__
	for (int i = 0; i < tblSize8; i++)
		pTable8[i] = powf(static_cast<float>(i) / maxVal, 2.19921875f);

	return;
}


void DeleteColorConvertTable(void) noexcept
{
	// nothing to do...
}

const float GetPowValue8u (const int& idx) noexcept
{
#ifdef _DEBUG
	if (idx < 0 || idx >= tblSize8)
		return 0.f;
#endif
	return pTable8[idx];
}