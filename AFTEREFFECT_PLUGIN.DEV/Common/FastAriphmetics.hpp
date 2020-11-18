#pragma once

inline double asqrt(const double& x)
{
	const double   xHalf = 0.50 * x;
	long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
	double         xRes = *(double*)&tmp;
	xRes *= (1.50 - (xHalf * xRes * xRes));
	return xRes * x;
}

inline float asqrt(const float& x)
{
	const float xHalf = 0.50f * x;
	int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
	float xRes = *(float*)&tmp;
	xRes *= (1.50f - (xHalf * xRes * xRes));
	return xRes * x;
}

