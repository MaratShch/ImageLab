#pragma once

#include "A.h"

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
