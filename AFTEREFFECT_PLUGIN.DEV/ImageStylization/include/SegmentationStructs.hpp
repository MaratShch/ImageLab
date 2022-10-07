#pragma once

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include <vector>

typedef struct CostData
{
	float cost;
	int32_t imin1;
	int32_t imin2;
	int32_t typemerging;

	CostData::CostData(void)
	{
		cost = 0.f;
		imin1 = imin2 = typemerging = 0;
	}
} CostData;

typedef struct Isegment
{
	std::vector<int32_t> pixels;
	std::vector<int32_t> srcIdx;
	float R;
	float G;
	float B;
	float Imin;
	float Imax;
} Isegment;

typedef struct Ssegment
{
	std::vector<int32_t> pixels;
	std::vector<int32_t> srcIdx;
	std::vector<Isegment> iSegments;
	float R;
	float G;
	float B;
	float Smin;
	float Smax;
} Ssegment;

//data structure to store the results of segmenting the Hue histogram
typedef struct Hsegment
{
	std::vector<int32_t> pixels;
	std::vector<int32_t> srcIdx;
	std::vector<Ssegment> sSegments;
	float R;
	float G;
	float B;
	float Hmin;
	float Hmax;
} Hsegment;


typedef struct dataRGB
{
	int32_t R;
	int32_t G;
	int32_t B;
};