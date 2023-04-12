#pragma once

#include "AE_Effect.h"
#include <cfloat>

#pragma pack(push)
#pragma pack(1)

typedef struct {
	A_u_char	B;
	A_u_char	G;
	A_u_char	R;
	A_u_char	A;
} PF_Pixel_BGRA_8u;

typedef struct {
	A_u_char	A;
	A_u_char	R;
	A_u_char	G;
	A_u_char	B;
} PF_Pixel_ARGB_8u;

typedef struct {
	A_u_short	B;
	A_u_short	G;
	A_u_short	R;
	A_u_short	A;
} PF_Pixel_BGRA_16u;

typedef struct {
	A_u_short	A;
	A_u_short	R;
	A_u_short	G;
	A_u_short	B;
} PF_Pixel_ARGB_16u;

typedef struct {
	PF_FpShort	B;
	PF_FpShort	G;
	PF_FpShort	R;
	PF_FpShort	A;
} PF_Pixel_BGRA_32f;

typedef struct {
	PF_FpShort	A;
	PF_FpShort	R;
	PF_FpShort	G;
	PF_FpShort	B;
} PF_Pixel_ARGB_32f;

typedef struct {
	A_u_char	V;
	A_u_char	U;
	A_u_char	Y;
	A_u_char	A;
} PF_Pixel_VUYA_8u;

typedef struct {
	A_u_short	V;
	A_u_short	U;
	A_u_short	Y;
	A_u_short	A;
} PF_Pixel_VUYA_16u;

typedef struct {
	PF_FpShort	V;
	PF_FpShort	U;
	PF_FpShort	Y;
	PF_FpShort	A;
} PF_Pixel_VUYA_32f;

typedef struct {
	A_u_long	_pad_ : 2;
	A_u_long	B : 10;
	A_u_long	G : 10;
	A_u_long	R : 10;
} PF_Pixel_RGB_10u;

#pragma pack(pop)

constexpr size_t PF_Pixel_BGRA_8u_size   = sizeof(PF_Pixel_BGRA_8u);
constexpr size_t PF_Pixel_ARGB_8u_size   = sizeof(PF_Pixel_ARGB_8u);
constexpr size_t PF_Pixel_BGRA_16u_size  = sizeof(PF_Pixel_BGRA_16u);
constexpr size_t PF_Pixel_ARGB_16u_size  = sizeof(PF_Pixel_ARGB_16u);
constexpr size_t PF_Pixel_BGRA_32f_size  = sizeof(PF_Pixel_BGRA_32f);
constexpr size_t PF_Pixel_ARGB_32f_size  = sizeof(PF_Pixel_ARGB_32f);
constexpr size_t PF_Pixel_VUYA_8u_size   = sizeof(PF_Pixel_VUYA_8u);
constexpr size_t PF_Pixel_VUYA_16u_size  = sizeof(PF_Pixel_VUYA_16u);
constexpr size_t PF_Pixel_VUYA_32f_size  = sizeof(PF_Pixel_VUYA_32f);
constexpr size_t PF_Pixel_RGB_10u_size   = sizeof(PF_Pixel_RGB_10u);

constexpr A_u_char u8_value_black = 0u;
constexpr A_u_char u8_value_white = 255u;
constexpr A_u_short u10_value_black = 0u;
constexpr A_u_short u10_value_white = 1023u;
constexpr A_u_short u16_value_black = 0u;
constexpr A_u_short u16_value_white = 32767u;
constexpr PF_FpShort f32_value_black = 0.f;
constexpr PF_FpShort f32_value_white = 1.0f - FLT_EPSILON;

#ifndef __NVCC__
#include "CommonPixFormatSFINAE.hpp"
#endif

