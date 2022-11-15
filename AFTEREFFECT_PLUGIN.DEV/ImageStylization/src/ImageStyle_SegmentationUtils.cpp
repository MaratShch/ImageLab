#include "ImageStylization.hpp"
#include "SegmentationStructs.hpp"
#include "SegmentationUtils.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"




inline std::vector<int32_t> ftc_utils_get_minima (const int32_t* __restrict pHist, const int32_t histSize) noexcept
{
	std::vector<int32_t> vectorOut;
	auto sPtype = std::make_unique<int32_t []>(histSize);

	if (sPtype)
	{
		auto type = sPtype.get();
		memset(type, 0, histSize * sizeof(type[0]));

		const int32_t lastSample = histSize - 1;
		int32_t next;
		int32_t i, j;

		type[0] = 5;
		type[lastSample] = 5;

		for (i = 1; i < lastSample; i++)
		{
			type[i] = 0;
			const int32_t diffprev = pHist[i] - pHist[i - 1];
			const int32_t diffnext = pHist[i] - pHist[i + 1];

			if ((diffprev < 0)  && (diffnext < 0))  type[i] = 1; /* minimum */
			if ((diffprev == 0) && (diffnext == 0)) type[i] = 4; /* flat    */
			if ((diffprev < 0)  && (diffnext == 0)) type[i] = 2; /* left    */
			if ((diffprev == 0) && (diffnext < 0))  type[i] = 3; /* right   */

		}

		/* check flat minima */
		for (i = 1; i < lastSample; i++)
		{
			if (2 == type[i])
			{
				//potential left endpoint of flat minimum
				//look for right endpoint
				for (j = i + 1; (j < lastSample) && (type[j] == 4); j++);
				if (3 == type[j])
				{
					//found right endpoint
					//mark center of flat zone as minimum
					type[(i + j) / 2] = 1;
				}
			}
		}

		vectorOut.push_back(0); //left endpoint
		for (i = 1; i < lastSample; i++) {
			if (1 == type[i])
			{
				vectorOut.push_back(i); //minimum
			}
		}
		vectorOut.push_back(lastSample); //right endpoint
	}

	return vectorOut;
}


inline std::vector<int32_t> ftc_utils_get_maxima(const int32_t* __restrict pHist, const int32_t histSize) noexcept
{
	std::vector<int> vectorOut;
	auto sPtype = std::make_unique<int32_t[]>(histSize);

	if (sPtype)
	{
		auto type = sPtype.get();
		memset(type, 0, histSize * sizeof(type[0]));

		const int32_t lastSample = histSize - 1;
		int32_t prev, curr, next;
		int32_t i, j;

		for (i = 1; i < lastSample; i++)
		{
			type[i] = 0;
			const int32_t diffprev = pHist[i] - pHist[i - 1];
			const int32_t diffnext = pHist[i] - pHist[i + 1];

			if ((diffprev > 0)  && (diffnext > 0))  type[i] = 1; //maximum
			if ((diffprev == 0) && (diffnext == 0)) type[i] = 4; //flat
			if ((diffprev > 0)  && (diffnext == 0)) type[i] = 2; //potential left endpoint of flat maximum
			if ((diffprev == 0) && (diffnext > 0))  type[i] = 3; //potential right endpoint of flat maximum 
		}

		/* check endpoints */
		type[0] = type[lastSample] = 0;
		if (pHist[0] > pHist[1]) type[0] = 1; /* maximum */
		if (pHist[0] == pHist[1]) type[0] = 2; /* potential left endpoint of flat maximum */
		if (pHist[lastSample] > pHist[lastSample - 1]) type[lastSample] = 1; /* maximum */
		if (pHist[lastSample] == pHist[lastSample - 1]) type[lastSample] = 3; /* potential right endpoint of flat maximum */

		for (i = 0; i < histSize; i++)
		{
			if (type[i] == 2)
			{ /* potential left endpoint of flat maximum
			  look for right endpoint */
				for (j = i + 1; (j < lastSample) && (type[j] == 4); j++);
				if (type[j] == 3)
				{
					/* found right endpoint
					mark center of flat zone as maximum */
					type[(i + j) / 2] = 1;
				}
			}
		}

		//output list of maxima
		for (i = 0; i < histSize; i++)
		{
			if (1 == type[i]) vectorOut.push_back(i);
		}
	} /* if (sPtype) */
	return vectorOut;
}



inline bool cost_already_computed(std::vector<CostData>& listCosts, const int32_t& imin1, const int32_t& imin2, CostData& cdata) noexcept
{
	const int32_t costsSize = static_cast<int32_t>(listCosts.size());
	bool found = false;

	for (int32_t k = 0; (k < costsSize) && (false == found); k++)
	{
		if ((listCosts[k].imin1 == imin1) && (listCosts[k].imin2 == imin2))
		{
			cdata = listCosts[k];
			found = true;
		}
	}

	return found;
}


void get_monotone_info(float* __restrict hist, int32_t* __restrict type, int32_t size, int& nincreasing, int& ndecreasing, bool extend_increasing) noexcept
{
	int32_t i = 0, j = 0;
	type[0] = nincreasing = ndecreasing = 0;

	for (i = 1; i < size; i++)
	{
		const float diffprev = hist[i] - hist[i - 1];
		type[i] = 0;
		if (diffprev > 0)
		{
			type[i] = 1;
			nincreasing++;
		}
		else
		{
			if (diffprev < 0)
			{
				type[i] = 2;
				ndecreasing++;
			}
		}
	}

	const int32_t typeV = extend_increasing ? 1 : 2;
	int32_t ilast = -1;
	for (i = 1; i < size; i++)
	{
		if (type[i] == typeV)
		{
			for (j = i - 1; (j > 0) && (type[j] == 0); j--) type[j] = typeV;
			ilast = i;
		}
	}
	if (ilast != -1)
	{
		for (j = ilast + 1; (j < size) && (type[j] == 0); j++)
			type[j] = typeV;
	}
	return;
}


inline void replace_monotone(float* __restrict hist, int32_t size, int32_t* __restrict type, bool replace_increasing) noexcept
{
	const int32_t& typeV = (true == replace_increasing ? 1 : 2);
	int32_t i = 0, j = 0;
	int32_t istart, iend;
	float cumulative;
	bool isfirst = true;

	while (i < size)
	{
		if (typeV == type[i])
		{
			if (isfirst)
			{
				istart = (i > 0) ? (i - 1) : i;
				isfirst = false;
			}
			if ((i == size - 1) || (type[i + 1] != typeV))
			{
				cumulative = 0.f;
				iend = i;
				for (j = istart; j <= iend; j++)
				{
					cumulative += hist[j];
				}

				cumulative /= (iend - istart + 1);

				for (j = istart; j <= iend; j++)
				{
					hist[j] = cumulative;
				}
				isfirst = true;
			}
		}
		i++;
	}

	return;
}


inline void pool_adjacent_violators(float* __restrict hist, float* __restrict hMono, int32_t size, bool increasing) noexcept
{
	CACHE_ALIGN int32_t Size[256]{};
	CACHE_ALIGN int32_t Type[256]{};
	int32_t nincreasing, ndecreasing;

	const size_t memSize = size * sizeof(float);
	memcpy(hMono, hist, memSize);

	if (increasing)
	{
		do {
			get_monotone_info(hMono, Type, size, nincreasing, ndecreasing, false);
			if (ndecreasing > 0)
				replace_monotone(hMono, size, Type, false);
		} while (ndecreasing > 0);
	}
	else
	{
		do {
			get_monotone_info(hMono, Type, size, nincreasing, ndecreasing, true);
			if (nincreasing > 0)
				replace_monotone(hMono, size, Type, true);
		} while (nincreasing > 0);
	}
	return;
}


inline float relative_entropy(const float& r, const float& p) noexcept
{
	float H;

	if (r == 0.f)
		H = -std::log(1.f - p);
	else if (r == 1.f)
		H = -std::log(p);
	else
		H = (r * std::log(r / p) + (1.f - r) * std::log((1.f - r) / (1.f - p)));

	return H;
}


inline float cost_monotone(const int32_t* __restrict hist0, int32_t i1, int32_t i2, bool increasing, float logeps) noexcept
{
	CACHE_ALIGN float hist[256]{};
	CACHE_ALIGN float hMono[256]{};

	const int32_t& L = i2 - i1 + 1;
	int32_t i, j;

	for (i = 0; i < L; i++)
		hist[i] = static_cast<float>(hist0[i1 + i]);

	//get monotone estimation
	pool_adjacent_violators(hist, hMono, L, increasing);

	//cumulated histograms
	for (i = 1; i < L; i++)
		hist[i] += hist[i - 1];

	for (i = 1; i < L; i++)
		hMono[i] += hMono[i - 1];

	const int32_t N = static_cast<int32_t>(hist[L - 1]);
	const float threshold = (std::log(static_cast<float>(L) * (L + 1) / 2) - logeps) / static_cast<float>(N);

	float H = 0.f, Hmax = FLT_MIN;

	for (i = 0; i < L; i++)
	{
		for (j = i; j < L; j++)
		{
			const float r = (0 == i ? hist[j] : hist[j] - hist[i - 1]) / static_cast<float>(N);
			const float p = (0 == i ? hMono[j] : hMono[j] - hMono[i - 1]) / static_cast<float>(N);

			H = relative_entropy(r, p);

			if (((i == 0) && (j == 0)) || (H > Hmax))
				Hmax = H;
		}
	}

	float const cost = static_cast<float>(N) * Hmax - (std::log(static_cast<float>(L*(L + 1) / 2)) - logeps);

	return cost;
}


CostData cost_merging
(
	const int32_t* __restrict hist,
	std::vector<CostData>& listCosts,
	std::vector<int>& separators,
	std::vector<int>& maxima,
	int i1,
	int i2,
	float logeps
) noexcept
{
	CostData cData;

	const float cost1 = cost_monotone(hist, separators[i1], maxima[i2 - 1], 1, logeps); //increasing
	const float cost2 = cost_monotone(hist, maxima[i1], separators[i2], 0, logeps); //decreasing

	if (cost1 < cost2)
	{
		cData.cost = cost1;
		cData.typemerging = 1;
	}
	else {
		cData.cost = cost2;
		cData.typemerging = 2;
	}
	cData.imin1 = separators[i1];
	cData.imin2 = separators[i2];

	listCosts.push_back(std::move(cData));

	return cData;
}


std::vector<int32_t> ftc_utils_segmentation (int32_t* inHist, int32_t inHistSize, float epsilon, bool isGray) noexcept
{
	int32_t* pHistogram = nullptr;
	const float fLogEps = std::log(epsilon);
	int32_t histSize = 0;
	int32_t i = 0, j = 0, n = 0;
	const bool circularHist = !isGray;

	const int32_t cyclicHistSize = inHistSize * 3;
	auto pCircularH = std::make_unique<int32_t[]>(cyclicHistSize);
	auto circularH = pCircularH.get();

	/* make circular histogram */
	if (false == isGray)
	{
		constexpr size_t histMemSize = sizeof(circularH);
		memset (circularH, 0, cyclicHistSize);

		for (i = 0; i < inHistSize; i++)
			circularH[i] = circularH[i + inHistSize] = circularH[i + 2 * inHistSize] = inHist[i];

		pHistogram = circularH;
		histSize = inHistSize * circular_size;
	}
	else
	{
		pHistogram = inHist;
		histSize = inHistSize;
	}

	std::vector<int32_t> SeparatorVector = ftc_utils_get_minima(pHistogram, histSize);
	std::vector<int32_t> MaximaVectorOut = ftc_utils_get_maxima(pHistogram, histSize);
	int32_t nIntervals = static_cast<int32_t>(SeparatorVector.size()) - 1;

	j = 1;
	std::vector<CostData> listCosts;

	while (nIntervals > j)
	{
		bool do_merging = true;
		bool costComputed = true;

		while (true == do_merging && nIntervals > j)
		{
			CostData cData;
			CostData cDataLowest;
			int32_t iLowest = -1;
			const int32_t nIntervalCnt = nIntervals - j;

			for (i = 0; i < nIntervalCnt; i++)
			{
				auto const i_j_sum = i + j + 1;
				auto const SeparatorIJ = SeparatorVector[i_j_sum];
				auto const SeparatorI = SeparatorVector[i];
				auto const SeparatorDiffs = SeparatorIJ - SeparatorI;

				if (circularHist && (SeparatorDiffs > histSize))
					continue;

				if (!(costComputed = cost_already_computed(listCosts, SeparatorI, SeparatorIJ, cData)))
				{
					cData = cost_merging(pHistogram, listCosts, SeparatorVector, MaximaVectorOut, i, i_j_sum, fLogEps);
				} /* if (false == (costComputed = cost_already_computed(listCost, listCostSize, SeparatorI, SeparatorJ, CData))) */

				if ((-1 == iLowest) || (cData.cost < cDataLowest.cost))
				{
					cDataLowest = cData;
					iLowest = i;
				}

			} /* for (i = 0; i < nIntervalMinusJ; i++) */

			  /* merge intervals with lowest cost, if it is smaller than 0 */
			if ((-1 != iLowest) && (cDataLowest.cost < 0.f))
			{
				/* remove minima with index ilowest+1 to ilowest+j */
				SeparatorVector.erase(SeparatorVector.begin() + iLowest + 1, SeparatorVector.begin() + iLowest + 1 + j);

				auto iterator2 = MaximaVectorOut.begin() + iLowest;
				//remove maxima associated to the removed minima
				if (1 == cDataLowest.typemerging)
					MaximaVectorOut.erase(iterator2, iterator2 + j);
				if (2 == cDataLowest.typemerging)
					MaximaVectorOut.erase(iterator2 + 1, iterator2 + j + 1);

				n = static_cast<int32_t>(SeparatorVector.size());
				nIntervals = n - 1;
			} /* if ((-1 != iLowest) && (CDataLowest.cost < 0)) */
			else
			{
				do_merging = false;
			}

		} /* while (true == do_merging && nIntervals > j) */

		j++; /* increase number of intervals to merge */

	} /* while (nIntervals > j) */

	if (false == isGray)
	{
		std::vector<int32_t> separatorsC;
		const int32_t doubleHistSize = 2 * inHistSize;

		for (j = 0; j < static_cast<int32_t>(SeparatorVector.size()); j++)
		{
			if ((SeparatorVector[j] >= inHistSize) && (SeparatorVector[j] < doubleHistSize))
			{
				separatorsC.push_back(SeparatorVector[j] - inHistSize);
			}
		}
		//if no separators then add endpoints
		if (separatorsC.size() == 0)
		{
			separatorsC.push_back(0);
			separatorsC.push_back(inHistSize - 1);
		}

		return separatorsC;
	} /* if (false == isGray) */   

	return SeparatorVector;
}


inline std::vector<Hsegment> hue_segmentation
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict pSrcImage,
	const A_long sizeX,
	const A_long sizeY,
	float Smin,
	int32_t nbins,
	float qH,
	std::vector<int32_t> ftcseg,
	const bool& optionGray = false
) noexcept
{
	std::vector<Hsegment>hSegments{};
	A_long i, j, n = 0;
	int32_t iH = 0;

#ifdef _DEBUG
	uint32_t loopDbgCnt = 0u;
#endif

	int32_t nsegments = static_cast<int32_t>(ftcseg.size());
	if ((nsegments == 2) && (ftcseg[0] == 0) && (ftcseg[1] == nbins - 1))
		nsegments = 1;

	auto idSegmentSPtr = std::make_unique<int32_t[]>(nbins);

	if (idSegmentSPtr)
	{
		auto idSegments = idSegmentSPtr.get();
		memset (idSegments, 0, nbins * sizeof(idSegments[0]));

		for (i = 0; i < nsegments; i++)
		{
			if (i < nsegments - 1)
			{
				for (int32_t k = ftcseg[i]; k < ftcseg[i + 1]; k++) idSegments[k] = i;
			}
			else
			{
				for (int32_t k = ftcseg[i]; k < nbins; k++) idSegments[k] = i;
				for (int32_t k = 0; k < ftcseg[0]; k++) idSegments[k] = i;
			}
		}

		for (i = 0; i < nsegments; i++)
		{
			Hsegment Hseg;
			Hseg.R = Hseg.G = Hseg.B = 0.f;
			Hseg.Hmin = ftcseg[i] * qH;
			Hseg.Hmax = (i < nsegments - 1) ? (ftcseg[i + 1] * qH) : (ftcseg[0] * qH);
			hSegments.push_back(std::move(Hseg));
		}

		//assign each pixel to one of the segments
		const A_long imgSize = sizeX * sizeY;
		for (n = 0; n < imgSize; n++)
		{
			iH = static_cast<int32_t>(hsi[n].H / qH);
			if (iH >= nbins)
				iH = nbins - 1;

			const auto& idseg = idSegments[iH];
			hSegments[idseg].pixels.push_back(n);
			hSegments[idseg].R += pSrcImage[n].R;
			hSegments[idseg].G += pSrcImage[n].G;
			hSegments[idseg].B += pSrcImage[n].B;
#ifdef _DEBUG
			loopDbgCnt++;
#endif
		}

		//Get average RGB values for each segment
		for (i = 0; i < nsegments; i++)
		{
			const float pSize = static_cast<float>(hSegments[i].pixels.size());
			hSegments[i].R = hSegments[i].R / pSize + 0.5f;
			hSegments[i].G = hSegments[i].G / pSize + 0.5f;
			hSegments[i].B = hSegments[i].B / pSize + 0.5f;
		}
	} /* if (idSegmentSPtr) */
	return hSegments;
}


inline void channel_segmentation_saturation
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict bgra,
	const A_long sizeX,
	const A_long sizeY,
	struct Hsegment& hSeg,
	int32_t nbins,
	float q,
	std::vector<int32_t> ftcseg
) noexcept
{
	int32_t i = 0, k = 0;

	auto idSegmentSPtr = std::make_unique<int32_t[]>(nbins);
	if (idSegmentSPtr)
	{
		auto idSegment = idSegmentSPtr.get();
		memset(idSegment, 0, nbins * sizeof(int32_t));

		const int32_t nsegments = static_cast<int32_t>(ftcseg.size()) - 1;

		//assign same ID to all bins belonging to same segment
		for (i = 0; i < nsegments; i++)
		{
			for (k = ftcseg[i]; k <= ftcseg[i + 1]; k++)
				idSegment[k] = i;
		}

		//store information of each segment in a data structure (for each Hsegment)
		for (i = 0; i < nsegments; i++)
		{
			Ssegment sSeg;
			sSeg.R = sSeg.G = sSeg.B = 0.f;
			sSeg.Smin = ftcseg[i] * q;
			sSeg.Smax = ftcseg[i + 1] * q;
			hSeg.sSegments.push_back(std::move(sSeg));
		}

		//assign each pixel to one of the segments
		for (k = 0; k < static_cast<int32_t>(hSeg.pixels.size()); k++)
		{
			auto const& n = hSeg.pixels[k];
			i = static_cast<int32_t>(hsi[n].S / q);

			if (i >= nbins)
				i = nbins - 1;

			const int32_t idSeg = idSegment[i];
			hSeg.sSegments[idSeg].pixels.push_back(n); //// !!!!!!!!!
			hSeg.sSegments[idSeg].R += bgra[n].R;
			hSeg.sSegments[idSeg].G += bgra[n].G;
			hSeg.sSegments[idSeg].B += bgra[n].B;
		}

		//Get average RGB values for each segment
		for (i = 0; i < static_cast<int32_t>(hSeg.sSegments.size()); i++)
		{
			const float pSize = static_cast<float>(hSeg.sSegments[i].pixels.size());
			hSeg.sSegments[i].R = hSeg.sSegments[i].R / pSize + 0.5f;
			hSeg.sSegments[i].G = hSeg.sSegments[i].G / pSize + 0.5f;
			hSeg.sSegments[i].B = hSeg.sSegments[i].B / pSize + 0.5f;
		}
	}
	return;
}


inline void channel_segmentation_intensity
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict bgra,
	const A_long sizeX,
	const A_long sizeY,
	struct Ssegment& Sseg,
	int nbins,
	float q,
	std::vector<int32_t> ftcseg
)
{
	int32_t i = 0, k = 0;

	auto idSegmentSPtr = std::make_unique<int32_t[]>(nbins);
	if (idSegmentSPtr)
	{
		auto idSegment = idSegmentSPtr.get();
		memset(idSegment, 0, nbins * sizeof(int32_t));

		const int32_t nsegments = ftcseg.size() - 1;

		//assign same ID to all bins belonging to same segment
		for (i = 0; i < nsegments; i++)
		{
			for (k = ftcseg[i]; k <= ftcseg[i + 1]; k++)
				idSegment[k] = i;
		}

		//store information of each segment in a data structure (for each Ssegment)
		for (i = 0; i < nsegments; i++)
		{
			Isegment Iseg;
			Iseg.R = Iseg.G = Iseg.B = 0.f;
			Iseg.Imin = static_cast<float>(ftcseg[i])     * q;
			Iseg.Imax = static_cast<float>(ftcseg[i + 1]) * q;
			Sseg.iSegments.push_back(std::move(Iseg));
		}

		//assign each pixel to one of the segments
		for (k = 0; k < static_cast<int32_t>(Sseg.pixels.size()); k++)
		{
			auto const& n = Sseg.pixels[k];

			i = static_cast<int32_t>(hsi[n].I / q);
			if (i >= nbins)
				i = nbins - 1;

			auto const& idseg = idSegment[i];

			Sseg.iSegments[idseg].pixels.push_back(n);
			Sseg.iSegments[idseg].R += bgra[n].R;
			Sseg.iSegments[idseg].G += bgra[n].G;
			Sseg.iSegments[idseg].B += bgra[n].B;
		} /* for (k = 0; k < static_cast<int32_t>(Sseg.pixels.size()); k++) */

		//Get average RGB values for each segment
		for (i = 0; i < static_cast<int32_t>(Sseg.iSegments.size()); i++)
		{
			const float& pixSize = static_cast<float>(Sseg.iSegments[i].pixels.size());
			Sseg.iSegments[i].R = (Sseg.iSegments[i].R / pixSize + 0.5f);
			Sseg.iSegments[i].G = (Sseg.iSegments[i].G / pixSize + 0.5f);
			Sseg.iSegments[i].B = (Sseg.iSegments[i].B / pixSize + 0.5f);
		}
	} /* if (idSegmentSPtr) */
	return;
}



std::vector<Hsegment> compute_color_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict imgSrc,
	const A_long sizeX,
	const A_long sizeY,
	float Smin,
	int32_t nbins,
	int32_t nbinsS,
	int32_t nbinsI,
	float qH,
	float qS,
	float qI,
	std::vector<int32_t> ftcseg,
	float eps
) noexcept
{
	std::vector<Hsegment> hSegments{};
	int32_t i, j, k, iS = 0;

	auto histSPtr = std::make_unique<int32_t[]>(nbinsS);
	auto histIPtr = std::make_unique<int32_t[]>(nbinsI);

	if (histSPtr && histIPtr)
	{
		hSegments  = hue_segmentation (hsi, imgSrc, sizeX, sizeY, Smin, nbins, qH, ftcseg);
		const int32_t nsegmentsH = static_cast<int32_t>(hSegments.size());

		auto histS = histSPtr.get();
		for (i = 0; i < nsegmentsH; i++)
		{
			memset (histS, 0, nbinsS * sizeof(histS[0]));
			const int32_t hSegPixSize = static_cast<const int32_t>(hSegments[i].pixels.size());
			for (k = 0; k < hSegPixSize; k++)
			{
				iS = static_cast<int32_t>(hsi[hSegments[i].pixels[k]].S / qS);
				if (iS >= nbinsS)
					iS = nbinsS - 1;
				histS[iS]++;
			}
			//segment saturation histogram
			std::vector<int32_t> ftcsegS = ftc_utils_segmentation (histS, nbinsS, eps, true);

			channel_segmentation_saturation (hsi, imgSrc, sizeX, sizeY, hSegments[i], nbinsS, qS, ftcsegS);

			auto histI = histIPtr.get();
			const int32_t sSegPixSize = static_cast<int32_t>(hSegments[i].sSegments.size());
			for (j = 0; j < sSegPixSize; j++)
			{
				memset (histI, 0, nbinsI * sizeof(histI[0]));
				const int32_t iSegPixSize = static_cast<const int32_t>(hSegments[i].sSegments[j].pixels.size());
				for (k = 0; k < iSegPixSize; k++)
				{
					int32_t iI = static_cast<int32_t>(hsi[hSegments[i].sSegments[j].pixels[k]].I / qI);
					if (iI >= nbinsI)
						iI = nbinsI - 1;
					histI[iI]++;
				}
					
				std::vector<int32_t> ftcsegI = ftc_utils_segmentation (histI, nbinsI, eps, true);
				channel_segmentation_intensity (hsi, imgSrc, sizeX, sizeY, hSegments[i].sSegments[j], nbinsI, qI, ftcsegI);
			}
		} /* for (i = 0; i < nsegmentsH; i++) */
	} /* if (histSPtr && histIPtr) */
	return hSegments;
}


void get_segmented_image
(
	std::vector<Isegment>& Isegments,
	std::vector<Hsegment>& Hsegments,
	const PF_Pixel_BGRA_8u* __restrict srcBgra,
	fDataRGB* __restrict fRGB,
	PF_Pixel_BGRA_8u* __restrict dstBgra,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept
{
	const A_long iSize = static_cast<A_long> (Isegments.size());
	const A_long hSize = static_cast<A_long> (Hsegments.size());
	const A_long imgSize = w * h;

	/* lets re-use temporary buffer for assemble output segmnented image */
	float* fR = reinterpret_cast<float*>(fRGB); 
	float* fG = fR + imgSize;
	float* fB = fG + imgSize;

	/* Get gray-level segmented image */
	for (A_long i = 0; i < iSize; i++)
	{
		const A_long iSegSize = static_cast<A_long>(Isegments[i].pixels.size());
		const float Rmean = Isegments[i].R;
		const float Gmean = Isegments[i].G;
		const float Bmean = Isegments[i].B;
		for (A_long n = 0; n < iSegSize; n++)
		{
			fR[Isegments[i].pixels[n]] = Rmean;
			fG[Isegments[i].pixels[n]] = Gmean;
			fB[Isegments[i].pixels[n]] = Bmean;
		} /* for (A_long n = 0; n < iSegSize; n++) */
	} /* for (A_long i = 0; i < iSize; i++) */

	/* Get color segmented image */
	for (A_long i = 0; i < hSize; i++)
	{
		const A_long sSize = static_cast<A_long>(Hsegments[i].sSegments.size());
		for (A_long j = 0; j < sSize; j++)
		{
			const A_long iSize = static_cast<A_long>(Hsegments[i].sSegments[j].iSegments.size());
			for (A_long k = 0; k < iSize; k++)
			{
				const float Rmean = Hsegments[i].sSegments[j].iSegments[k].R;
				const float Gmean = Hsegments[i].sSegments[j].iSegments[k].G;
				const float Bmean = Hsegments[i].sSegments[j].iSegments[k].B;
				const A_long pSize = static_cast<A_long>(Hsegments[i].sSegments[j].iSegments[k].pixels.size());
				
				for (A_long n = 0; n < pSize; n++)
				{
					fR[Hsegments[i].sSegments[j].iSegments[k].pixels[n]] = Rmean;
					fG[Hsegments[i].sSegments[j].iSegments[k].pixels[n]] = Gmean;
					fB[Hsegments[i].sSegments[j].iSegments[k].pixels[n]] = Bmean;
				}
			}
		}
	} /* for (A_long i = 0; i < hSize; i++) */

	/* store assembled segmented image */
	for (A_long j = 0; j < h; j++)
	{
		const PF_Pixel_BGRA_8u* pSrcLine = srcBgra + j * srcPitch;
		PF_Pixel_BGRA_8u* pDstLine = dstBgra + j * dstPitch;
		const A_long fTmpStorageIdx = j * w;

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < w; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].R = static_cast<A_u_char>(CLAMP_VALUE(fR[fTmpStorageIdx + i], 0.f, 255.f));
			pDstLine[i].G = static_cast<A_u_char>(CLAMP_VALUE(fG[fTmpStorageIdx + i], 0.f, 255.f));
			pDstLine[i].B = static_cast<A_u_char>(CLAMP_VALUE(fB[fTmpStorageIdx + i], 0.f, 255.f));
		}
	}

	return;
}


std::vector<Isegment> compute_gray_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict imgSrc,
	const A_long sizeX,
	const A_long sizeY,
	float Smin,
	int32_t nbinsI,
	float qI,
	std::vector<int32_t> ftcsegI
) noexcept
{
	std::vector<Isegment> iSegments;
	auto sIdsegmentsI = std::make_unique<int32_t []>(nbinsI);
	int ii, n;

	if (sIdsegmentsI)
	{
		auto idsegmentsI = sIdsegmentsI.get();
		const int32_t nsegmentsI = ftcsegI.size() - 1;

		for (ii = 0; ii < nsegmentsI; ii++)
		{
			for (int32_t k = ftcsegI[ii]; k <= ftcsegI[ii + 1]; k++)
				idsegmentsI[k] = ii;
		} /* for (int32_t ii = 0; ii < nsegmentsI; ii++) */

		for (ii = 0; ii < nsegmentsI; ii++)
		{
			Isegment Iseg;
			Iseg.R = Iseg.G = Iseg.B = 0.f;
			Iseg.Imin = ftcsegI[ii] * qI;
			Iseg.Imax = ftcsegI[ii + 1] * qI;
			iSegments.push_back(std::move(Iseg));
		} /* for (ii = 0; ii < nsegmentsI; ii++) */

		const A_long imgSize = sizeX * sizeY;
		for (n = 0; n < imgSize; n++)
		{
			auto const& S = hsi[n].S;
			if (S <= Smin)
			{
				int32_t iI = static_cast<int32_t>(hsi[n].I / qI);
				if (iI >= nbinsI)
					iI = nbinsI - 1;

				const int32_t idseg = idsegmentsI[iI];
				iSegments[idseg].pixels.push_back(n);
				iSegments[idseg].R += imgSrc[n].R;
				iSegments[idseg].G += imgSrc[n].G;
				iSegments[idseg].B += imgSrc[n].B;
			}
		} /* for (n = 0; n < imgSize; n++) */

		for (int32_t i = 0; i < nsegmentsI; i++)
		{
			const float pSize = static_cast<float>(iSegments[i].pixels.size());
			const float Rmean = iSegments[i].R / pSize + 0.5f;
			const float Gmean = iSegments[i].G / pSize + 0.5f;
			const float Bmean = iSegments[i].B / pSize + 0.5f;

			iSegments[i].R = iSegments[i].G = iSegments[i].B = (Rmean + Gmean + Bmean) / 3.f + 0.5f;
		} /* for (int32_t i = 0; i < nsegmentsI; i++) */
	} /* if (sIdsegmentsI) */

	return iSegments;
}