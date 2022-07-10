#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"


namespace ArtMosaic
{
	using PixelPos = A_long;

	class Pixel
	{
	public:
		PixelPos x, y;
		Pixel(const PixelPos& x0, const PixelPos& y0) noexcept
		{
			x = x0;
			y = y0;
		}
		Pixel() noexcept
		{
			Pixel(0, 0);
		}
	};


	class Color
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
	};


	template <typename T>
	inline constexpr T sq(const T& x) noexcept
	{
		return (x * x);
	}

	inline constexpr float color_distance(const Color& c1, const Color& c2) noexcept
	{
		return (sq(c1.b - c2.b) + sq(c1.g - c2.g) + sq(c1.r - c2.r));
	}



	class Superpixel
	{
		public:

		explicit Superpixel(const A_long& x0, const A_long& y0, const Color& c) noexcept
		{
			col = c;
			x = static_cast<float>(x0);
			y = static_cast<float>(y0);
			size = 0;
			pix = nullptr;
		}

//		explicit Superpixel(const A_long& x0, const A_long y0, const Color&& c) noexcept
//		{
//			col = c;
//			x = static_cast<float>(x0);
//			y = static_cast<float>(y0);
//			size = 0;
//			pix = nullptr;
//		}


		float distance (const A_long& i, const A_long& j, const Color& c, const float& wSpace) const noexcept
		{
			const float eucldist = sq(x - static_cast<float>(i)) + sq(y - static_cast<float>(j));
			const float colordist = color_distance (col, c);
			return wSpace * wSpace * eucldist + colordist;
		}

		Pixel* pix;
		Color col;
		float x, y;
		int size;

	};

	void fillProcBuf (Color* pBuf, const A_long& pixNumber, const float& val) noexcept;
	void fillProcBuf (std::unique_ptr<Color[]>& pBuf, const A_long& pixNumber, const float& val) noexcept;




	inline bool isInside
	(
		const A_long& x,
		const A_long& y,
		const A_long& w,
		const A_long& h
	) noexcept 
	{
		return (0 <= x && x < w && 0 <= y && y < h);
	}

	inline bool isInside
	(
		const Pixel& p,
		const A_long& w,
		const A_long& h
	) noexcept
	{
		return isInside (p.x, p.y, w, h);
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline Color getSrcPixel
	(
		const T* __restrict I,
		const A_long& x,
		const A_long& y,
		const A_long pitch
	) noexcept
	{
		const T& srcPixel = I[x + y * pitch];
		Color pix (srcPixel.R, srcPixel.G, srcPixel.B);
		return pix;
	}

	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline Color getSrcPixel
	(
		const T* __restrict I,
		const Pixel& p,
		const A_long pitch
	) noexcept
	{
		const T& srcPixel = I[p.x + p.y * pitch];
		Color c (srcPixel.R, srcPixel.G, srcPixel.B);
		return c;
	}




	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline A_long sqNormGradient
	(
		const T* __restrict I,
		const Pixel& p,
		const A_long& w,
		const A_long& h,
		const A_long& pitch
	) noexcept
	{
		A_long g = 0;
		Pixel q = p;
		++q.x;
		if (!isInside(q, w, h))
			q.x = p.x - 1;
		if (!isInside(q, w, h))
			q.x = p.x; // case w=1
		{
			Color c1 = getSrcPixel(I, q, pitch);
			Color c2 = getSrcPixel(I, p, pitch);
			g += static_cast<A_long>(color_distance(c1, c2));
		}

		q = p;
		++q.y;
		if (isInside(q, w, h))
			q.y = p.y - 1;
		if (isInside(q, w, h))
			q.y = p.y; // case h=1
		{
			Color c1 = getSrcPixel(I, q, pitch);
			Color c2 = getSrcPixel(I, p, pitch);
			g += static_cast<A_long>(color_distance(c1, c2));
		}
		return g;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline void moveMinimalGradient
	(
		std::vector<Superpixel>& sp,
		const T* __restrict I,
		const A_long& radius,
		const A_long& K,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		for (A_long k = 0; k < K; k++)
		{
			A_long j, i;
			A_long minNorm = std::numeric_limits<A_long>::max();

			const A_long x = static_cast<A_long>(sp[k].x);
			const A_long y = static_cast<A_long>(sp[k].y);

			for (j = -radius; j <= radius; j++)
			{
				for (i = -radius; i <= radius; i++)
				{
					Pixel p(x + i, y + j);
					if (isInside(p, sizeX, sizeY))
					{
						const A_long g = sqNormGradient (I, p, sizeX, sizeY, pitch);
						if (g < minNorm)
						{
							sp[k].x = static_cast<float>(p.x);
							sp[k].y = static_cast<float>(p.y);
							minNorm = g;
						} /* if (g < minNorm) */

					} /* if (isInside(p, sizeX, sizeY)) */

				} /* for (i = -radius; i <= radius; i++) */

			} /* for (j = -radius; j <= radius; j++) */

		} /* for (A_long k = 0; k < K; k++) */

		return;
	}

	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	void assignmentStep
	(
		std::vector<Superpixel>& sp,
		const T* __restrict pSrc,
		const float& wSpace,
		const A_long& S,
		std::unique_ptr<Color[]>& l,
		std::unique_ptr<Color[]>& d,
		const A_long sizeX,
		const A_long sizeY,
		const A_long pitch
	)
	{
		A_long i, j;
		const A_long size = static_cast<A_long>(sp.size());
		for (A_long k = 0; k < size; k++)
		{
			for (i = -S; i < S; i++)
			{
				for (j = -S; j < S; j++)
				{
					A_long ip = static_cast<int>((sp[k].x + 0.50f) + i);
					A_long jp = static_cast<int>((sp[k].y + 0.50f) + j);

					if (isInside(ip, jp, sizeX, sizeY))
					{
						const Color color = getSrcPixel(pSrc, ip, jp, pitch);
						float dist = sp[k].distance (ip, jp, color, wSpace);
						
						//if (d(ip, jp) > dist)
						//{
						//	d(ip, jp) = dist;
						//	l(ip, jp) = k;
						//}

					}
				} /* for (int j = -S; j < S; j++) */

			} /* for (int i = -S; i < S; i++) */

		} /* for (A_long k = 0; k < size; k++) */
		return;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline std::vector<Superpixel> SlicImageImpl
	(
		const T* __restrict pSrc,
		std::unique_ptr<Color[]>& pOut,
		const Color& GrayColor,
		const float& m,
		A_long& K,
		const A_long& g,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& srcPitch
	) noexcept
	{
		std::vector<Superpixel> sp;
		A_long j, i;

		/* init superpixel */
		const float superPixInitVal = static_cast<float>(sizeX * sizeY) / static_cast<float>(K);
		const A_long S  = FastCompute::Max(1, static_cast<A_long>(FastCompute::Sqrt(superPixInitVal)));
		const A_long nX = FastCompute::Max(1, sizeX / S);
		const A_long nY = FastCompute::Max(1, sizeY / S);

		const A_long padw = FastCompute::Max(0, sizeX - S * nX);
		const A_long padh = FastCompute::Max(0, sizeY - S * nY);
		const A_long s = S >> 1;

		const A_long halfPadW = padw >> 1;
		const A_long halfPadH = padh >> 1;

		for (j = 0; j < nY; j++)
		{
			const A_long jj = j * S + s + halfPadH;
			for (i = 0; i < nX; i++)
			{
				const A_long ii = i * S + s + halfPadW;
				if (isInside(ii, jj, sizeX, sizeY))
				{
					const T& srcPix = pSrc[jj * srcPitch + ii];
					Color color(static_cast<float>(srcPix.R), static_cast<float>(srcPix.G), static_cast<float>(srcPix.B));
					sp.push_back(Superpixel(ii, jj, color));
				}
			}
		}
		K = static_cast<A_long>(sp.size());

		moveMinimalGradient (sp, pSrc, g, K, sizeX, sizeY, srcPitch);

		const float wSpace = m / static_cast<float>(S);
		std::vector<float> il(6, 0);
		std::vector<std::vector<float>> centers(sp.size(), il);

		constexpr A_long MaxIterSlic = 1000;
		constexpr float RmseMax = 0.5f;
		constexpr float InitialError = 2.f * RmseMax + 1.f;
		float E = InitialError;

		const A_long procBufSize = sizeX * sizeY;
		const A_long bytesSize = procBufSize * sizeof(Color);
		auto procBuf = std::make_unique<Color[]>(procBufSize);

		if (procBuf)
		{
			for (i = 0; i < MaxIterSlic && E > RmseMax; i++)
			{
				fillProcBuf (procBuf, procBufSize, std::numeric_limits<float>::max());
				assignmentStep (sp, pSrc, wSpace, S, pOut, procBuf,sizeX, sizeY, srcPitch);
			} /* for (i = 0; i < MaxIterSlic && E > RmseMax; i++) */
		}
		
		return sp;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline bool SlicImage
	(
		const T* __restrict pSrc,
		      T* __restrict pDst,
		const Color& grayColor,
		const float& m,
		      A_long& k,
		const A_long& g,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& srcPitch,
		const A_long& dstPitch
	) noexcept
	{
		int GridStep = 0;
		const A_long bufSize = CreateAlignment(sizeX * sizeY, CACHE_LINE);
		auto pOut = std::make_unique<Color[]>(bufSize);

		bool bResult = false;

		if (pOut)
		{
			std::vector<Superpixel> sp = SlicImageImpl(pSrc, pOut, grayColor, m, k, g, sizeX, sizeY, srcPitch);
			bResult = true;
		}

		return bResult;
	}

}; /* ArtMosaic */
