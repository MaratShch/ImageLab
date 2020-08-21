#pragma once

typedef struct _PixelBGRA_u8
{
	csSDK_uint8 B;
	csSDK_uint8 G;
	csSDK_uint8 R;
	csSDK_uint8 A;
}PixelBGRA_u8;

typedef struct _PixelBGRA_u16
{
	csSDK_uint16 B;
	csSDK_uint16 G;
	csSDK_uint16 R;
	csSDK_uint16 A;
}PixelBGRA_u16;

typedef struct _PixelBGRA_f32
{
	float B;
	float G;
	float R;
	float A;
}PixelBGRA_f32;

typedef struct _PixelARGB_u8
{
	csSDK_uint8 A;
	csSDK_uint8 R;
	csSDK_uint8 G;
	csSDK_uint8 B;
}PixelARGB_u8;

typedef struct _PixelARGB_u16
{
	csSDK_uint16 A;
	csSDK_uint16 R;
	csSDK_uint16 G;
	csSDK_uint16 B;
}PixelARGB_u16;

typedef struct _PixelARGB_f32
{
	float A;
	float R;
	float G;
	float B;
}PixelARGB_f32;


typedef struct _PixelYUVA_u8
{
	csSDK_uint8 Y;
	csSDK_uint8 U;
	csSDK_uint8 V;
	csSDK_uint8 A;
}PixelYUVA_u8;

typedef struct _PixelYUVA_u16
{
	csSDK_uint16 Y;
	csSDK_uint16 U;
	csSDK_uint16 V;
	csSDK_uint16 A;
}PixelYUVA_u16;

typedef struct _PixelYUVA_f32
{
	float Y;
	float U;
	float V;
	float A;
}PixelYUVA_f32;

