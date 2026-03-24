#pragma once

#include <stack>
#include <queue>
#include <memory>
#include <limits>
#include <iostream>

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "FastAriphmetics.hpp"
#include "CommonPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"
#include "ClassRestrictions.hpp"
#include "MosaicMemHandler.hpp"

#ifndef TRUE
 #define TRUE 1
#endif

inline void CheckPlanarRange(const float* buffer, int32_t numPixels, const char* planeName)
{
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();

    for (int32_t i = 0; i < numPixels; i++)
    {
        if (buffer[i] < minVal) minVal = buffer[i];
        if (buffer[i] > maxVal) maxVal = buffer[i];
    }

    // Replace printf with your plugin framework's native logging if needed
    std::cout << "Plane " << planeName << " MIN = " << minVal << " MAX = " << maxVal << std::endl;
}

inline void CheckPlanarRange(const int32_t* buffer, int32_t numPixels, const char* planeName)
{
    int32_t minVal = std::numeric_limits<int32_t>::max();
    int32_t maxVal = std::numeric_limits<int32_t>::lowest();

    for (int32_t i = 0; i < numPixels; i++)
    {
        if (buffer[i] < minVal) minVal = buffer[i];
        if (buffer[i] > maxVal) maxVal = buffer[i];
    }

    // Replace printf with your plugin framework's native logging if needed
    std::cout << "Plane " << planeName << " MIN = " << minVal << " MAX = " << maxVal << std::endl;
}

void MosaicAlgorithmMain (const MemHandler& memHndl, A_long width, A_long height, A_long K = 1000);
 
namespace ArtMosaic
{
	using PixelPos = A_long;

	class Pixel final
	{
	public:
		PixelPos x, y;
		Pixel (const PixelPos& x0, const PixelPos& y0) noexcept
		{
			x = x0;
			y = y0;
		}
		Pixel() noexcept
		{
			Pixel(0, 0);
		}

		~Pixel() = default;

		inline const A_long getIdx (const A_long pitch) const noexcept
		{
			return x + y * pitch;
		}
	};


	class Color final
	{
	public:
		float r, g, b;

		Color() noexcept { r = g = b = 0; }

		Color(const float& r0, const float& g0, const float& b0) noexcept
		{
			r = r0;
			g = g0;
			b = b0;
		}

		~Color() = default;
	};


	void fillProcBuf   (Color* pBuf, const A_long pixNumber, const float val) noexcept;
	void fillProcBuf   (std::unique_ptr<Color[]>& pBuf,  const A_long pixNumber, const float val) noexcept;
	void fillProcBuf   (A_long* pBuf, const A_long pixNumber, const A_long val) noexcept;
	void fillProcBuf   (std::unique_ptr<A_long[]>& pBuf, const A_long pixNumber, const A_long val) noexcept;
	void fillProcBuf   (float* pBuf, const A_long pixNumber, const float val) noexcept;
	void fillProcBuf   (std::unique_ptr<float[]>& pBuf, const A_long pixNumber, const float val) noexcept;


}; // ArtMosaic/