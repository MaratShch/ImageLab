#pragma once

#include "SegmentationStructs.hpp"
#include "CommonPixFormat.hpp"
#include "ImageAuxPixFormat.hpp"

std::vector<int32_t> ftc_utils_segmentation(const int32_t* inHist, const int32_t& inHistSize, float epsilon, bool circularHist) noexcept;

std::vector<Hsegment> compute_color_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const PF_Pixel_BGRA_8u* __restrict bgra,
	float Smin,
	int32_t nbins,
	int32_t nbinsS,
	int32_t nbinsI,
	float qH,
	float qS,
	float qI,
	std::vector<int32_t>& ftcseg,
	int32_t w,
	int32_t h,
	float eps
) noexcept;