#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageMosaicUtils.hpp"



void ArtMosaic::fillProcBuf (Color* pBuf, const A_long& pixNumber, const float& val) noexcept
{
	constexpr A_long elemInStruct = sizeof(pBuf[0]) / sizeof(pBuf[0].r);
	const A_long rawSize    = pixNumber * elemInStruct;
	const A_long rawSize24  = rawSize / 24;
	const A_long rawFract24 = rawSize % 24;

	float* pBufF = reinterpret_cast<float*>(pBuf);

	const __m256 fPattern = _mm256_set_ps(val, val, val, val, val, val, val, val);
	for (A_long i = 0; i < rawSize24; i++)
	{
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
	}

	for (A_long i = 0; i < rawFract24; i++)
		pBufF[i] = val;

	return;
}

void ArtMosaic::fillProcBuf (std::unique_ptr<Color[]>& pBuf, const A_long& pixNumber, const float& val) noexcept
{
	ArtMosaic::fillProcBuf (pBuf.get(), pixNumber, val);
}

void ArtMosaic::fillProcBuf (A_long* pBuf, const A_long& pixNumber, const A_long& val) noexcept
{
	const A_long rawSize16  = pixNumber / 16;
	const A_long rawFract16 = pixNumber % 16;
	const __m256i iPattern = _mm256_set_epi32 (val, val, val, val, val, val, val, val);
	__m256i* pBufAvxPtr = reinterpret_cast<__m256i*>(pBuf);

	for (A_long i = 0; i < rawSize16; i++)
	{
		_mm256_store_si256 (pBufAvxPtr, iPattern), pBufAvxPtr++;
		_mm256_store_si256 (pBufAvxPtr, iPattern), pBufAvxPtr++;
	}

	if (0 != rawFract16)
	{
		A_long* pBufPtr = reinterpret_cast<A_long*>(pBufAvxPtr);
		for (A_long i = 0; i < rawFract16; i++)
			pBufPtr[i] = val;
	}

	return;
}

void ArtMosaic::fillProcBuf(std::unique_ptr<A_long[]>& pBuf, const A_long& pixNumber, const A_long& val) noexcept
{
	ArtMosaic::fillProcBuf (pBuf.get(), pixNumber, val);
}


void ArtMosaic::fillProcBuf (float* pBuf, const A_long& pixNumber, const float& val) noexcept
{
	const A_long rawSize16 = pixNumber / 16;
	const A_long rawFract16 = pixNumber % 16;
	const __m256 fPattern = _mm256_set_ps(val, val, val, val, val, val, val, val);
	__m256* pBufAvxPtr = reinterpret_cast<__m256*>(pBuf);

	for (A_long i = 0; i < rawSize16; i++)
	{
		_mm256_store_ps (reinterpret_cast<float*>(pBufAvxPtr), fPattern), pBufAvxPtr++;
		_mm256_store_ps (reinterpret_cast<float*>(pBufAvxPtr), fPattern), pBufAvxPtr++;
	}

	if (0 != rawFract16)
	{
		float* pBufPtr = reinterpret_cast<float*>(pBufAvxPtr);
		for (A_long i = 0; i < rawFract16; i++)
			pBufPtr[i] = val;
	}

	return;
}


void ArtMosaic::fillProcBuf(std::unique_ptr<float[]>& pBuf, const A_long& pixNumber, const float& val) noexcept
{
	ArtMosaic::fillProcBuf(pBuf.get(), pixNumber, val);
}


float ArtMosaic::computeError (const std::vector<ArtMosaic::Superpixel>& sp, const std::vector<std::vector<float>>& centers) noexcept
{
	float E{ 0 };
	const A_long K = static_cast<A_long>(sp.size());
	for (A_long k = 0; k < K; k++)
		E += (centers[k][0] - sp[k].x) * (centers[k][0] - sp[k].x) + (centers[k][1] - sp[k].y) * (centers[k][1] - sp[k].y);

	return FastCompute::Sqrt(E / static_cast<float>(K));
}


void ArtMosaic::moveCenters (std::vector<ArtMosaic::Superpixel>& sp, const std::vector<std::vector<float>>& centers) noexcept
{
	const A_long K = static_cast<A_long>(sp.size());
	for (A_long k = 0; k < K; k++)
	{
		sp[k].x = centers[k][0];
		sp[k].y = centers[k][1];
		ArtMosaic::Color col(centers[k][2], centers[k][3], centers[k][4]);
		sp[k].col = col;
		sp[k].size = static_cast<A_long>(centers[k][5]);
	}
	return;
}


ArtMosaic::Pixel ArtMosaic::neighbor (const ArtMosaic::PixelPos& i, const ArtMosaic::PixelPos& j, const A_long& n) noexcept
{
	if (n >= 0 && n < 4)
	{
		ArtMosaic::Pixel p(i, j);

		switch (n)
		{
			case 0: ++p.x; break;
			case 1: --p.y; break;
			case 2: --p.x; break;
			case 3: ++p.y; break;
		}

		return p;
	}

	return ArtMosaic::Pixel();
}


ArtMosaic::Pixel ArtMosaic::neighbor (const ArtMosaic::Pixel& p, const A_long& n) noexcept
{
	return ArtMosaic::neighbor (p.x, p.y, n);
}


void ArtMosaic::labelCC
(
	std::unique_ptr<A_long[]>& CC,
	std::vector<int32_t>& H,
	std::unique_ptr<A_long[]>& L,
	const A_long& sizeX,
	const A_long& sizeY
) noexcept
{
	std::stack<ArtMosaic::Pixel> S;
	auto cc = CC.get();
	auto l  = L.get();

	const A_long size = sizeX * sizeY;
	ArtMosaic::fillProcBuf(cc, size, -1);

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long lineIdx = j * sizeX;
		for (A_long i = 0; i < sizeY; i++)
		{
			const A_long idx = i + lineIdx;
			if (-1 != cc[idx])
				continue;

			S.push(ArtMosaic::Pixel(i, j));

			const A_long label = l[idx];
			const A_long labelcc = static_cast<A_long>(H.size());
			cc[idx] = labelcc;
			H.push_back(0);

			while (!S.empty())
			{
				++H.back();
				ArtMosaic::Pixel p = S.top();
				S.pop();
				for (A_long n = 0; n < 4; n++)
				{
					ArtMosaic::Pixel q = ArtMosaic::neighbor (p, n);
					const A_long qIdx = q.x + q.y * sizeX;
					if (isInside(q, sizeX, sizeY) && -1 == cc[qIdx] && label == l[qIdx])
					{
						S.push(q);
						cc[qIdx] = labelcc;
					}
				}
			} /* while (!S.empty()) */

		} /* for (A_long i = 0; i < sizeY; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}


void ArtMosaic::discardMinorCC
(
	std::unique_ptr<A_long[]>& CC,
	const std::vector<int>& H,
	std::unique_ptr<A_long[]>& L,
	const A_long& K,
	const A_long& sizeX,
	const A_long& sizeY
) noexcept
{
	A_long j, i;
	const A_long w = sizeX, h = sizeY;
	std::vector<A_long> maxSizeCC (K, -1); // superpixel -> label of largest cc

	auto cc = CC.get();
	auto l = L.get();
	
	for (j = 0; j < sizeY; j++) // Fill maxSizeCC
	{
		const A_long line_idx = j * sizeX;
		for (i = 0; i < sizeX; i++)
		{
			const A_long idx = line_idx + i;
			const A_long labelCC = cc[idx];
			const A_long labelS = l[idx];
			if (labelS >= 0)
			{
				A_long& s = maxSizeCC[labelS];
				if (s < 0 || H[s] < H[labelCC])
					s = labelCC;
			} /* if (labelS >= 0) */
		} /* for (i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	for (j = 0; j < sizeY; j++) // Make orphans the minor cc of each superpixel
	{
		const A_long line_idx = j * sizeX;
		for (i = 0; i < sizeX; i++)
		{
			const A_long idx = line_idx + i;
			auto ll = l[idx];
			if (l[idx] >= 0 && cc[idx] != maxSizeCC[ll])
			{
				cc[idx] = -1;
				l [idx] = -1;
			}
		} /* for (i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}



