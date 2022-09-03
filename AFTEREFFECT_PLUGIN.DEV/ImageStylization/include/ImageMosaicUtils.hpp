#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"
#include <stack>
#include <queue>

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

		inline const A_long getIdx (const A_long& pitch) const noexcept
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

		explicit Superpixel (const A_long& x0, const A_long& y0, const Color& c) noexcept
		{
			x = static_cast<float>(x0);
			y = static_cast<float>(y0);
			size = 0;
			col = c;
			pix = nullptr;
		}

		explicit Superpixel (const A_long& x0, const A_long y0, const Color&& c) noexcept
		{
			x = static_cast<float>(x0);
			y = static_cast<float>(y0);
			size = 0;
			col = c;
			pix = nullptr;
		}


		float distance (const A_long& i, const A_long& j, const Color& c, const float& wSpace) const noexcept
		{
			const float eucldist = sq(x - static_cast<float>(i)) + sq(y - static_cast<float>(j));
			const float colordist = color_distance (col, c);
			return wSpace * wSpace * eucldist + colordist;
		}

		float x, y;
		A_long size;
		Color col;
		Pixel* pix;
	};

	void fillProcBuf   (Color* pBuf, const A_long& pixNumber, const float& val) noexcept;
	void fillProcBuf   (std::unique_ptr<Color[]>& pBuf,  const A_long& pixNumber, const float& val) noexcept;
	void fillProcBuf   (A_long* pBuf, const A_long& pixNumber, const A_long& val) noexcept;
	void fillProcBuf   (std::unique_ptr<A_long[]>& pBuf, const A_long& pixNumber, const A_long& val) noexcept;
	void fillProcBuf   (float* pBuf, const A_long& pixNumber, const float& val) noexcept;
	void fillProcBuf   (std::unique_ptr<float[]>& pBuf, const A_long& pixNumber, const float& val) noexcept;

	float computeError (const std::vector<Superpixel>& sp, const std::vector<std::vector<float>>& centers) noexcept;
	void  moveCenters  (std::vector<Superpixel>& sp, const std::vector< std::vector<float>>& centers) noexcept;

	Pixel neighbor (const PixelPos& i, const PixelPos& j, const A_long& n) noexcept;
	Pixel neighbor (const Pixel& p, const A_long& n) noexcept;

	void labelCC (std::unique_ptr<A_long[]>& CC, std::vector<int32_t>& H, std::unique_ptr<A_long[]>& L, const A_long& sizeX, const A_long& sizeY) noexcept;
	void discardMinorCC (std::unique_ptr<A_long[]>& CC, const std::vector<int>& H, std::unique_ptr<A_long[]>& L, const A_long& K, const A_long& sizeX, const A_long& sizeY) noexcept;

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
		const A_long x,
		const A_long y,
		const A_long pitch
	) noexcept
	{
		const T& srcPixel = I[x + y * pitch];
		Color pix (static_cast<float>(srcPixel.R), static_cast<float>(srcPixel.G), static_cast<float>(srcPixel.B));
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
		auto const& x = p.x;
		auto const& y = p.y;
		return getSrcPixel (I, x, y, pitch);
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
		std::unique_ptr<A_long[]>& L,
		std::unique_ptr<float []>& D,
		const A_long sizeX,
		const A_long sizeY,
		const A_long pitch
	)
	{
#ifdef _DEBUG
		uint32_t dbgCount = 0u;
#endif

		auto l = L.get();
		auto d = D.get();

		A_long i = 0, j = 0, k = 0;
		const A_long size = static_cast<A_long>(sp.size());

		for (k = 0; k < size; k++)
		{
			for (i = -S; i < S; i++)
			{
				for (j = -S; j < S; j++)
				{
					A_long ip = static_cast<A_long>(sp[k].x + 0.50f) + i;
					A_long jp = static_cast<A_long>(sp[k].y + 0.50f) + j;

					if (isInside(ip, jp, sizeX, sizeY))
					{
						const Color color = getSrcPixel(pSrc, ip, jp, pitch);
						float dist = sp[k].distance (ip, jp, color, wSpace);
						const A_long idx = jp * sizeX + ip;
						if (d[idx] > dist)
						{
							d[idx] = dist;
							l[idx] = k;
#ifdef _DEBUG
							dbgCount++;
#endif
						}
					}
				} /* for (int j = -S; j < S; j++) */

			} /* for (int i = -S; i < S; i++) */

		} /* for (A_long k = 0; k < size; k++) */
		return;
	}

	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	void updateStep
	(
		std::vector<std::vector<float>>& centers,
		std::unique_ptr<A_long[]>& L,
		const T* __restrict pSrc,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		A_long i, j;

		auto l = L.get();
		const A_long K = static_cast<A_long>(centers.size());

		for (A_long kk = 0; kk < K; kk++)
		{
			for (A_long ll = 0; ll < 6; ll++)
			{
				centers[kk][ll] = 0;
			}
		}

		for (j = 0; j < sizeY; j++)
		{
			for (i = 0; i < sizeX; i++)
			{
				const A_long tmpIdx = i + j * sizeX;
				const A_long srcIdx = i + j * pitch;

				if (l[tmpIdx] < 0)
					continue;

				/* because source pixel defined as templated parameter - let's convert R,G,B component to common type cover all possible pixel types */
				const float pix[5] = { 
					static_cast<float>(i),
					static_cast<float>(j),
					static_cast<float>(pSrc[srcIdx].R),
					static_cast<float>(pSrc[srcIdx].G),
					static_cast<float>(pSrc[srcIdx].B)
				};

				auto const lIdx = l[tmpIdx];
				std::vector<float>& c = centers[lIdx];
				for (A_long s = 0; s < 5; s++)
					c[s] += pix[s];
				++c[5];
			}
		}

		for (A_long kk = 0; kk < K; kk++)
		{
			if (centers[kk][5] > 0)
			{
				for (A_long ss = 0; ss < 5; ss++)
					centers[kk][ss] /= centers[kk][5];
			}
		}
		return;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	void computeSuperpixelColors
	(
		const T* __restrict pSrc,
		std::vector<ArtMosaic::Superpixel>& sp,
		std::unique_ptr<A_long[]>& L,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		A_long j, i, k;
		const A_long size = sizeX * sizeY;
		const A_long K = static_cast<const A_long>(sp.size());
		const A_long bufSize = K * 4;

		auto col = std::make_unique<int[]>(bufSize);
		if (col)
		{
			fillProcBuf(col, bufSize, 0);
			auto l = L.get();
			auto const _col = col.get();

			for (j = 0; j < sizeY; j++)
			{
				for (i = 0; i < sizeX; i++)
				{
					const A_long idx = j * sizeX + i;
					const A_long k = l[idx];
					if (k >= 0)
					{
						const T& colorPixel = pSrc[j * pitch + i];
						_col[k * 4 + 0] += colorPixel.R;
						_col[k * 4 + 1] += colorPixel.G;
						_col[k * 4 + 2] += colorPixel.B;
						_col[k * 4 + 3] ++;
					} /* if (k >= 0) */

				} /* for (i = 0; i < sizeX; i++) */
			} /* for (j = 0; j < sizeY; j++) */

			for (k = 0; k < K; k++)
			{
				const auto& col0 = _col[k * 4 + 0];
				const auto& col1 = _col[k * 4 + 1];
				const auto& col2 = _col[k * 4 + 2];
				const auto& col3 = _col[k * 4 + 3];

				if (col3 > 0)
				{
					sp[k].col = Color(col0 / col3, col1 / col3, col2 / col3);
				} /* if (col3 > 0) */
			} /* for (k = 0; k < K; k++) */

		} /* if (col) */

		return;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	void assignOrphans
	(
		const T* __restrict pSrc,
		const std::vector<ArtMosaic::Superpixel>& sp,
		std::unique_ptr<A_long[]>& L,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		A_long i, j, dist, n;
		std::queue<ArtMosaic::Pixel> Q;
		auto l = L.get();
		constexpr A_long cMinVal{ std::numeric_limits<A_long>::min() };

		for (j = 0; j < sizeY; j++)
		{
			for (i = 0; i < sizeX; i++)
			{
				const A_long idx = j * sizeX + i;
				if (l[idx] < 0)
					l[idx] = cMinVal;
			} /* for (i = 0; i < sizeX; i++) */
		} /* for (j = 0; j < sizeY; j++) */

		for (dist = -1; TRUE; --dist)
		{
			size_t Qsize = Q.size();
			for (j = 0; j < sizeY; j++)
			{
				for (i = 0; i < sizeX; i++)
				{
					const A_long idx = j * sizeX + i;
					if (l[idx] > dist)
					{
						for (A_long n = 0; n < 4; n++)
						{
							Pixel q = neighbor(i, j, n);
							const A_long lIdx = q.getIdx(sizeX);
							if (isInside(q, sizeX, sizeY) && l[lIdx] < dist)
							{
								l[lIdx] = dist;
								Q.push(q);
							}
						} /* for (A_long n = 0; n < 4; n++) */

					} /* if (l[idx] > dist) */
				} /* for (i = 0; i < sizeX; i++) */
			} /* for (j = 0; j < sizeY; j++) */

			if (Qsize == Q.size())
				break;
			Qsize = Q.size();
		} /* for (dist = -1; TRUE; --dist) */

		while (!Q.empty())
		{ 
			Pixel p = Q.front();
			Q.pop();
			A_long nearest = -1;
			float minDist = std::numeric_limits<float>::max();

			for (n = 0; n < 4; n++)
			{ 
				Pixel q = neighbor(p, n);
				const A_long idx = q.getIdx(sizeX);
				if (!isInside(q, sizeX, sizeY) || l[idx])
					continue;

				const T& srcPixel = pSrc[p.getIdx(pitch)];
				const Color col1(static_cast<float>(srcPixel.R), static_cast<float>(srcPixel.G), static_cast<float>(srcPixel.B));
				const Color col2 = sp[l[q.getIdx(sizeX)]].col;

				float dist = color_distance(col1, col2);
				if (dist < minDist)
				{
					minDist = dist;
					nearest = l[q.getIdx(sizeX)];
				}
			} /* for (n = 0; n < 4; n++) */

			l[p.getIdx(sizeX)] = nearest;
		} /* while (!Q.empty()) */

		return;
	}


	inline bool enforceConnectivity
	(
		std::vector<Superpixel>& sp,
		std::unique_ptr<A_long[]>& L,
		const PF_Pixel_ARGB_32f* __restrict pSrc,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		std::vector<int32_t> H;

		return true;
	}

	inline bool enforceConnectivity
	(
		std::vector<Superpixel>& sp,
		std::unique_ptr<A_long[]>& L,
		const PF_Pixel_BGRA_32f* __restrict pSrc,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		std::vector<int32_t> H;

		return true;
	}




	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	inline bool enforceConnectivity
	(
		std::vector<Superpixel>& sp,
		std::unique_ptr<A_long []>& L,
		const T* __restrict pSrc,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& pitch
	) noexcept
	{
		std::vector<int32_t> H;
		auto CC = std::make_unique<A_long[]>(sizeX * sizeY);
		bool retVal = false;

		if (CC)
		{
			labelCC (CC, H, L, sizeX, sizeY);

			const A_long K = static_cast<const A_long>(sp.size());
			discardMinorCC (CC, H, L, K, sizeX, sizeY);
			computeSuperpixelColors (pSrc, sp, L, sizeX, sizeY, pitch);
			assignOrphans (pSrc, sp, L, sizeX, sizeY, pitch);

			retVal = true;
		}

		return retVal;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	std::vector<Superpixel> SlicImageImpl
	(
		const T* __restrict pSrc,
		std::unique_ptr<A_long[]>& L,
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
		bool bVal = false;

#ifdef _DEBUG
		volatile uint32_t dbgLoopCnt = 0u;
		volatile size_t spSize = 0ull;
		volatile size_t centerSize = 0ull;
#endif

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
					sp.push_back(Superpixel(ii, jj, std::move(color)));
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
		constexpr float floatMaxVal = std::numeric_limits<float>::max();
		float E = InitialError;

		const A_long procBufSize = sizeX * sizeY;
		auto D = std::make_unique<float []>(procBufSize);

		if (D)
		{
			for (i = 0; i < MaxIterSlic && E > RmseMax; i++)
			{
#ifdef _DEBUG
				dbgLoopCnt++;
				spSize = sp.size();
				centerSize = centers.size();
#endif
				fillProcBuf (D, procBufSize, floatMaxVal);
				assignmentStep (sp, pSrc, wSpace, S, L, D ,sizeX, sizeY, srcPitch);
				updateStep (centers, L, pSrc, sizeX, sizeY, srcPitch);
				E = computeError (sp, centers);
				moveCenters (sp, centers);
			} /* for (i = 0; i < MaxIterSlic && E > RmseMax; i++) */

			bVal = enforceConnectivity(sp, L, pSrc, sizeX, sizeY, srcPitch);
		} /* if (procBuf && D && L) */
		
		if (false == bVal)
			sp.clear();

		return sp;
	}


	template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
	void slic_output
	(
		const T* __restrict pSrc,
		      T* __restrict pDst,
		const std::vector<Superpixel>& sp,
		      std::unique_ptr<A_long[]>& L,
		const Color& col,
		const A_long& sizeX,
		const A_long& sizeY,
		const A_long& srcPitch,
		const A_long& dstPitch,
		bool borders = true
	)
	{
		return;
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
		const A_long bufSize = sizeX * sizeY;
		auto L    = std::make_unique<A_long []>(bufSize);
		size_t vectorSize = 0;
		bool bResult = false;

		if (L)
		{
			std::vector<Superpixel> sp = SlicImageImpl(pSrc, L, grayColor, m, k, g, sizeX, sizeY, srcPitch);
			if (0 != (vectorSize = sp.size()))
			{
				/* SLIC OUTPUT */
				slic_output (pSrc, pDst, sp, L, grayColor, sizeX, sizeY, srcPitch, dstPitch, true);

				bResult = true;
			}
		}

		return bResult;
	}


	template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
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
		const A_long bufSize = CreateAlignment(sizeX * sizeY, CACHE_LINE);
		auto pOut = std::make_unique<Color[]>(bufSize);

//		std::vector<Superpixel> sp = SlicImageImpl (pSrc, pOut, grayColor, m, k, g, sizeX, sizeY, srcPitch);
		return false;
	}

}; /* ArtMosaic */
