#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"


namespace ArtMosaic
{
	using PixelPos = A_long;


	template <typename T>
	inline constexpr T sq (const T& x) noexcept
	{
		return (x * x);
	}

	template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline constexpr float color_distance(const T& c1, const T& c2) noexcept
	{
		return (static_cast<float>(sq(c1.B - c2.B) + sq(c1.G - c2.G) + sq(c1.R - c2.R)));
	}


	class Pixel
	{
	public:
		PixelPos x, y;
		Pixel(const PixelPos& x0, const PixelPos& y0)
		{
			x = x0;
			y = y0;
		}
		Pixel()
		{ 
			Pixel(0, 0);
		}
	};


	template <typename T>
	class Color
	{
	public:
		T r, g, b;

		Color() { ; }
		Color(const T& r0, const T& g0, const T& b0)
		{
			r = r0;
			g = g0;
			b = b0;
		}
	};


	template <typename T>
	class Superpixel
	{
		public:

			explicit Superpixel (const A_long x0, const A_long y0, const T& c) 
			{
				pix = nullptr;
				col = c;
				x = static_cast<float>(x0);
				y = static_cast<float>(y0);
				size = 0;
			}

			explicit Superpixel(const A_long x0, const A_long y0, const T&& c)
			{
				pix = nullptr;
				col = c;
				x = static_cast<float>(x0);
				y = static_cast<float>(y0);
				size = 0;
			}

//			float distance (const A_long& i, const A_long& j, const Color<U>& c, const float& wSpace) const noexcept
//			{
//				const float eucldist = sq(x - static_cast<float>(i)) + sq(y - static_cast<float>(j));
//				const float colordist = color_distance (col, c);
//				return wSpace * wSpace * eucldist + colordist;
//			}

			Pixel* pix;
			T col;
			float x, y;
			int size;

	};
	
	inline bool isInside
	(
		const int& x, 
		const int& y,
		const int& w,
		const int& h
	) noexcept 
	{
		return (0 <= x && x < w && 0 <= y && y < h);
	}

	inline bool isInside
	(
		const Pixel& p,
		const int& w,
		const int& h
	) noexcept
	{
		return isInside (p.x, p.y, w, h);
	}

#if 0
	int sqNormGradient(const Image<Color>& I, const Pixel& p) {
		int g = 0;
		Pixel q = p;
		++q.x;
		if (!I.inside(q))
			q.x = p.x - 1;
		if (!I.inside(q))
			q.x = p.x; // case w=1
		g += color_dist(I(q), I(p));

		q = p;
		++q.y;
		if (!I.inside(q))
			q.y = p.y - 1;
		if (!I.inside(q))
			q.y = p.y; // case h=1
		g += color_dist(I(q), I(p));
		return g;
	}
#endif

	template <typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline void moveMinimalGradient
	(
		std::vector<Superpixel<T>>& sp,
		const T* __restrict I,
		const A_long& radius,
		const A_long& K,
		const A_long& sizeX,
		const A_long& sizeY
	) noexcept
	{
		for (A_long k = 0; k < K; k++)
		{
			int j, i;
			A_long minNorm = std::numeric_limits<A_long>::max();

			const int x = static_cast<int>(sp[k].x);
			const int y = static_cast<int>(sp[k].y);

			for (j = -radius; j <= radius; j++)
			{
				for (i = -radius; i <= radius; i++)
				{
					Pixel p(x + i, y + j);
					if (isInside(p, sizeX, sizeY))
					{
						A_long g = 0;// sqNormGradient(I, p); !!!!
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


	template <typename U, typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline std::vector<Superpixel<T>> SimpleLinearIterativeClustering
	(
		const T* __restrict pSrc,
		std::unique_ptr<Color<U>[]>& pOut,
		const float& m,
		      A_long& K,
		const A_long& g,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& srcPitch
	) noexcept
	{
		auto pOutBuffer{ pOut.get() };
		std::vector<Superpixel<T>> sp;

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
					sp.push_back(Superpixel<T>(ii, jj, pSrc[jj * srcPitch + ii]));
			}
		}
		K = static_cast<A_long>(sp.size());

		moveMinimalGradient (sp, pSrc, g, K, sizeX, sizeY);

		return sp;
	}


	template <typename U, typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline bool SlicImage
	(
		const T* __restrict pSrc,
		      T* __restrict pDst,
		const Color<U>& GrayColor,
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
		const A_long frameSize = CreateAlignment (sizeX * sizeY, CACHE_LINE);
		auto tmpOut = std::make_unique<Color<U>[]>(frameSize);

		std::vector<Superpixel<T>> sp = SimpleLinearIterativeClustering (pSrc, tmpOut, m, k, g, sizeX, sizeY, srcPitch);

		return true;
	}

}; /* ArtMosaic */
