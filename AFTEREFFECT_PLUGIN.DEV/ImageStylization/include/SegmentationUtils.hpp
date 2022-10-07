#pragma once

#include "SegmentationStructs.hpp"
#include "CommonPixFormat.hpp"
#include "ImageAuxPixFormat.hpp"

std::vector<int32_t> ftc_utils_segmentation(const int32_t* inHist, const int32_t& inHistSize, float epsilon, bool circularHist) noexcept;

std::vector<Hsegment> compute_color_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const PF_Pixel_BGRA_8u* __restrict bgra,
	const A_long width,
	const A_long height,
	const A_long srcPitch,
	const A_long tmpPitch,
	float Smin,
	int32_t nbins,
	int32_t nbinsS,
	int32_t nbinsI,
	float qH,
	float qS,
	float qI,
	std::vector<int32_t>& ftcseg,
	float eps
) noexcept;

void get_list_grays_colors
(
	std::vector<Isegment>& Isegments,
	std::vector<Hsegment>& Hsegments,
	std::vector<dataRGB>& meanRGB_I,
	std::vector<dataRGB>& meanRGB_H,
	std::vector<dataRGB>& meanRGB_HS,
	std::vector<dataRGB>& meanRGB_HSI,
	std::vector<int32_t>& icolorsH,
	std::vector<int32_t>& icolorsS
) noexcept;

void get_segmented_image
(
	std::vector<Isegment> Isegments,
	std::vector<Hsegment> Hsegments,
	PF_Pixel_BGRA_8u* __restrict bgra,
	int32_t w,
	int32_t h,
	int32_t pitch
) noexcept;