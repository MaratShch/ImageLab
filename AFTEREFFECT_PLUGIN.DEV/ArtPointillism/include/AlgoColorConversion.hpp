#include <cmath>
#include <algorithm>
#include "FastAriphmetics.hpp"

// Scalar Helper: sRGB [0..1] -> CIELab [L, a, b]
// Used during Painter Initialization
inline void Convert_RGB_to_Lab_Scalar(float r, float g, float b, float& L, float& a_out, float& b_out)
{
    // 1. Gamma Correction (sRGB -> Linear)
    auto srgb_to_lin = [](float v)
	{
        return (v <= 0.04045f) ? (v / 12.92f) : FastCompute::Pow((v + 0.055f) / 1.055f, 2.4f);
    };

    float r_lin = srgb_to_lin(r);
    float g_lin = srgb_to_lin(g);
    float b_lin = srgb_to_lin(b);

    // 2. Linear RGB -> XYZ (D65)
    float X = 0.4124564f * r_lin + 0.3575761f * g_lin + 0.1804375f * b_lin;
    float Y = 0.2126729f * r_lin + 0.7151522f * g_lin + 0.0721750f * b_lin;
    float Z = 0.0193339f * r_lin + 0.1191920f * g_lin + 0.9503041f * b_lin;

    // 3. XYZ -> Lab
    const float Xn = 0.95047f;
    const float Yn = 1.00000f;
    const float Zn = 1.08883f;

    float xr = X / Xn;
    float yr = Y / Yn;
    float zr = Z / Zn;

    auto f = [](float t)
	{
        return (t > 0.008856f) ? FastCompute::Cbrt(t) : (7.787f * t + 16.0f/116.0f);
    };

    float fx = f(xr);
    float fy = f(yr);
    float fz = f(zr);

    L = (116.0f * fy) - 16.0f;
    a_out = 500.0f * (fx - fy);
    b_out = 200.0f * (fy - fz);
}