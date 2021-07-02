#pragma once

#pragma pack(push)
#pragma pack(1)

typedef struct
{
	float H;
	float S;
	float I;
} PF_Pixel_HSI_32f;

#pragma pack(pop)

constexpr size_t PF_Pixel_HSI_32f_size = sizeof(PF_Pixel_HSI_32f);