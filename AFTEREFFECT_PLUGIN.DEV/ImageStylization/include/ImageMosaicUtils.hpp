#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"


namespace ArtMosaic
{
	using PixelPos = A_long;

	typedef struct ProcPixel
	{
		float R;
		float G;
		float B;
	}ProcPixel;

	void fillProcBuf (ProcPixel* pBuf, const A_long& pixNumber, const float& val) noexcept;
	void fillProcBuf (std::unique_ptr<ProcPixel[]>& pBuf, const A_long& pixNumber, const float& val) noexcept;

	template <typename T>
	inline constexpr T sq (const T& x) noexcept
	{
		return (x * x);
	}

	template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline constexpr float color_distance(const T& c1, const T& c2) noexcept
	{
		return (static_cast<float>(sq(c1.b - c2.b) + sq(c1.g - c2.g) + sq(c1.r - c2.r)));
	}


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


	template <typename T>
	class Color
	{
	public:
		T r, g, b;

		Color() noexcept { ; }
		Color(const T& r0, const T& g0, const T& b0) noexcept
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

			explicit Superpixel (const A_long x0, const A_long y0, const T& c) noexcept
			{
				pix = nullptr;
				col = c;
				x = static_cast<float>(x0);
				y = static_cast<float>(y0);
				size = 0;
			}

			explicit Superpixel(const A_long x0, const A_long y0, const T&& c) noexcept
			{
				pix = nullptr;
				col = c;
				x = static_cast<float>(x0);
				y = static_cast<float>(y0);
				size = 0;
			}

//			float distance (const A_long& i, const A_long& j, const Color<float>& c, const float& wSpace) const noexcept;
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

	template <typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline Color<float> getSrcPixel
	(
		const T* __restrict I,
		const A_long& x,
		const A_long& y,
		const A_long pitch
	) noexcept
	{
		const T& srcPixel = I[x + y * pitch];
		Color<float> pix(srcPixel.R, srcPixel.G, srcPixel.B);
		return pix;
	}

	template <typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline Color<float> getSrcPixel
	(
		const T* __restrict I,
		const Pixel& p,
		const A_long pitch
	) noexcept
	{
		const T& srcPixel = I[p.x + p.y * pitch];
		Color<float> c(srcPixel.R, srcPixel.G, srcPixel.B);
		return c;
	}


	template <typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
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
			Color<float> c1 = getSrcPixel(I, q, pitch);
			Color<float> c2 = getSrcPixel(I, p, pitch);
			g += static_cast<A_long>(color_distance(c1, c2));
		}

		q = p;
		++q.y;
		if (isInside(q, w, h))
			q.y = p.y - 1;
		if (isInside(q, w, h))
			q.y = p.y; // case h=1
		{
			Color<float> c1 = getSrcPixel(I, q, pitch);
			Color<float> c2 = getSrcPixel(I, p, pitch);
			g += static_cast<A_long>(color_distance(c1, c2));
		}
		return g;
	}


	template <typename T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
	inline void moveMinimalGradient
	(
		std::vector<Superpixel<T>>& sp,
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
						const A_long g = sqNormGradient(I, p, sizeX, sizeY, pitch);
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
	void assignmentStep
	(
		std::vector<Superpixel<T>>& sp,
		const T* __restrict pSrc,
		const float& wSpace, 
		const A_long& S,
		std::unique_ptr<Color<U>[]>& l,
		std::unique_ptr<ProcPixel[]>& d,
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
						const Color<float> color = getSrcPixel(pSrc, ip, jp, pitch);
//						float dist = sp[k].dist5D (ip, jp, color, wSpace);

					}
				} /* for (int j = -S; j < S; j++) */

			} /* for (int i = -S; i < S; i++) */

		} /* for (A_long k = 0; k < size; k++) */
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

		moveMinimalGradient (sp, pSrc, g, K, sizeX, sizeY, srcPitch);
	
		const float wSpace = m / static_cast<float>(S); 
		std::vector<float> il(6, 0); 
		std::vector<std::vector<float>> centers(sp.size(), il);

		constexpr A_long MaxIterSlic = 1000;
		constexpr float RmseMax = 0.5f;
		constexpr float InitialError = 2.f * RmseMax + 1.f;
		float E = InitialError;

		const A_long procBufSize = sizeX * sizeY;
		const A_long bytesSize = procBufSize * sizeof(ProcPixel);
		auto procBuffer = std::make_unique<ProcPixel[]>(procBufSize);

		if (procBuffer)
		{
			constexpr float fillValue = std::numeric_limits<float>::max();

			for (i = 0; i < MaxIterSlic && E > RmseMax; i++)
			{
				fillProcBuf (procBuffer, procBufSize, fillValue);
				assignmentStep (sp, pSrc, wSpace, S, pOut, procBuffer, sizeX, sizeY, srcPitch);
			} /* for (i = 0; i < MaxIterSlic && E > RmseMax; i++) */

		} /* if (procBuffer) */

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

		if (tmpOut)
		{
			std::vector<Superpixel<T>> sp = SimpleLinearIterativeClustering(pSrc, tmpOut, m, k, g, sizeX, sizeY, srcPitch);
		}

		return true;
	}

}; /* ArtMosaic */
