#include "ImageStylization.hpp"
#include "SegmentationStructs.hpp"
#include "SegmentationUtils.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"


inline std::vector<int> ftc_utils_get_minima(const int32_t* __restrict pHist, const int32_t histSize) noexcept
{
	CACHE_ALIGN int type[256 /*hist_size_H * circular_size*/]{};
	std::vector<int> vectorOut{};

	const int32_t lastSample = histSize - 1;
	int32_t prev, curr, next;
	int32_t i, j;

	type[0] = 5;
	__VECTOR_ALIGNED__
	prev = pHist[0];
	for (i = 1; i < lastSample; i++)
	{
		curr = pHist[i];
		next = pHist[i + 1];

		type[i] = 0;
		const int32_t diffprev = curr - prev;
		const int32_t diffnext = curr - next;

		if (diffprev < 0)
		{
			if (diffnext < 0)
				type[i] = 1; /* minimum */
			else if (diffnext == 0)
				type[i] = 2; /* potential left endpoint of flat minimum */
		}
		else if (diffprev == 0)
		{
			if (diffnext < 0)
				type[i] = 3; /* potential right endpoint of flat minimum */
			else if (diffnext == 0)
				type[i] = 4; /* flat */
		}

		prev = curr;
	}
	type[lastSample] = 5;

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
				type[(i + j) >> 1] = 1;
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
	return vectorOut;
}


inline std::vector<int> ftc_utils_get_maxima(const int32_t* __restrict pHist, const int32_t histSize) noexcept
{
	CACHE_ALIGN int32_t type[256 /*hist_size_H * circular_size*/]{};
	std::vector<int> vectorOut{};

	const int32_t& lastSample = histSize - 1;
	int32_t prev, curr, next;
	int32_t i, j;

	type[0] = 0;
	__VECTOR_ALIGNED__
		prev = pHist[0];
	for (i = 1; i < lastSample; i++)
	{
		curr = pHist[i];
		next = pHist[i + 1];

		type[i] = 0;
		const int32_t diffprev = curr - prev;
		const int32_t diffnext = curr - next;

		if (diffprev > 0)
		{
			if (diffnext > 0)
				type[i] = 1; /* maximum */
			else if (diffnext == 0)
				type[i] = 2; /* potential left endpoint of flat maximum */
		}
		else if (diffprev == 0)
		{
			if (diffnext > 0)
				type[i] = 3; /* potential right endpoint of flat maximum  */
		}

		prev = curr;
	}
	type[lastSample] = 0;

	/* check endpoints */
	if (pHist[0] >  pHist[1]) type[0] = 1; /* maximum */
	if (pHist[0] == pHist[1]) type[0] = 2; /* potential left endpoint of flat maximum */
	if (pHist[lastSample]  > pHist[lastSample - 1]) type[lastSample] = 1; /* maximum */
	if (pHist[lastSample] == pHist[lastSample - 1]) type[lastSample] = 3; /* potential right endpoint of flat maximum */

	for (i = 0; i < histSize; i++)
	{
		if (type[i] == 2)
		{ /* potential left endpoint of flat maximum
		  look for right endpoint */
			for (j = i + 1; (j < histSize - 1) && (type[j] == 4); j++);
			if (type[j] == 3)
			{
				/* found right endpoint
				mark center of flat zone as maximum */
				type[(i + j) >> 1] = 1;
			}
		}
	}

	//output list of maxima
	for (i = 0; i < histSize; i++)
	{
		if (1 == type[i])
		{
			/* maximum */
			vectorOut.push_back(i);
		}
	}
	return vectorOut;
}



inline bool cost_already_computed(std::vector<CostData>& listCosts, const int32_t& imin1, const int32_t& imin2, CostData& cdata) noexcept
{
	const int32_t& costsSize = static_cast<int32_t>(listCosts.size());
	bool found = false;

	__VECTOR_ALIGNED__
		for (int32_t k = 0; k < costsSize && false == found; k++)
		{
			auto const& iMin1 = listCosts[k].imin1;
			auto const& iMin2 = listCosts[k].imin2;

			if ((iMin1 == imin1) && (iMin2 == imin2))
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

	__VECTOR_ALIGNED__
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

	float const cost = static_cast<float>(N) * Hmax - (std::log(static_cast<float>(L*(L + 1) >> 1)) - logeps);

	return cost;
}


inline CostData cost_merging(
	const int32_t* __restrict hist,
	std::vector<CostData>& listCosts,
	std::vector<int>& separators,
	std::vector<int>& maxima,
	int i1,
	int i2,
	float logeps) noexcept
{
	CostData cData{};

	auto const& separatorsI1 = separators[i1];
	auto const& separatorsI2 = separators[i2];
	auto const& maximaI2 = maxima[i2 - 1];

	const float& cost1 = cost_monotone(hist, separatorsI1, maximaI2, 1, logeps); //increasing
	const float& cost2 = cost_monotone(hist, separatorsI1, separatorsI2, 0, logeps); //decreasing

	if (cost1 < cost2) {
		cData.cost = cost1;
		cData.typemerging = 1;
	}
	else {
		cData.cost = cost2;
		cData.typemerging = 2;
	}
	cData.imin1 = separators[i1];
	cData.imin2 = separators[i2];

	listCosts.push_back(cData);

	return cData;
}


std::vector<int32_t> ftc_utils_segmentation(const int32_t* inHist, const int32_t& inHistSize, float epsilon, bool isGray) noexcept
{
	CACHE_ALIGN int32_t circularH[hist_size_H * circular_size];
	int32_t* pHistogram = nullptr;
	const float& fLogEps = log(epsilon);
	int32_t histSize;
	const bool& circularHist = !isGray;

	/* make circular histogram */
	if (false == isGray)
	{
		const int32_t inHistDblSize = 2 * inHistSize;
		memset(circularH, 0, sizeof(circularH));
		__VECTOR_ALIGNED__
			for (int i = 0; i < inHistSize; i++)
			{
				circularH[i] = inHist[i];
				circularH[i + inHistSize] = inHist[i];
				circularH[i + inHistDblSize] = inHist[i];
			}
		pHistogram = circularH;
		histSize = inHistSize * circular_size;
	}
	else
	{
		pHistogram = const_cast<int32_t*>(inHist);
		histSize = inHistSize;
	}

	std::vector<int32_t> separatorsC;
	std::vector<int> SeparatorVector = ftc_utils_get_minima(pHistogram, histSize);
	std::vector<int> MaximaVectorOut = ftc_utils_get_maxima(pHistogram, histSize);
	int32_t nIntervals = static_cast<int32_t>(SeparatorVector.size()) - 1;

	int32_t j = 1, i;
	std::vector<CostData> listCosts;

	while (nIntervals > j)
	{
		bool do_merging = true;
		bool costComputed = false;

		while (true == do_merging && nIntervals > j)
		{
			CACHE_ALIGN CostData cData;
			CACHE_ALIGN CostData cDataLowest;
			int32_t iLowest = -1;
			const int32_t nIntervalCnt = nIntervals - j;

			for (i = 0; i < nIntervalCnt; i++)
			{
				auto const i_j_sum = i + j + 1;
				auto const SeparatorJ = SeparatorVector[i_j_sum];
				auto const SeparatorI = SeparatorVector[i];
				auto const SeparatorDiffs = SeparatorJ - SeparatorI;

				if (circularHist && SeparatorDiffs > histSize)
					continue;

				if (false == (costComputed = cost_already_computed(listCosts, SeparatorI, SeparatorJ, cData)))
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
			if ((-1 != iLowest) && (cDataLowest.cost < 0))
			{
				/* remove minima with index ilowest+1 to ilowest+j */
				SeparatorVector.erase(SeparatorVector.begin() + iLowest + 1, SeparatorVector.begin() + iLowest + j + 1);
				auto& iterator = MaximaVectorOut.begin() + iLowest;
				//remove maxima associated to the removed minima
				if (1 == cDataLowest.typemerging)
				{
					MaximaVectorOut.erase(iterator, iterator + j);
				}
				if (2 == cDataLowest.typemerging)
				{
					MaximaVectorOut.erase(iterator + 1, iterator + j + 1);
				}

				nIntervals = static_cast<int32_t>(SeparatorVector.size()) - 1;
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
		const int32_t doubleHistSize = 2 * inHistSize;

		for (j = 0; j < SeparatorVector.size(); j++)
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
	}

	return (false == isGray ? separatorsC : SeparatorVector);
}


template <typename T>
inline std::vector<Hsegment> hue_segmentation
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const T*                __restrict bgra,
	float Smin,
	int32_t nbins,
	float qH,
	std::vector<int32_t>& ftcseg,
	int32_t w,
	int32_t h,
	const bool& optionGray = false
) noexcept
{
	CACHE_ALIGN int32_t idSegments[128]{};
	std::vector<Hsegment>hSegments;
	int32_t nsegments = static_cast<int32_t>(ftcseg.size());
	int32_t i, k, n;
	int32_t iH;
	const int32_t imgSize = w * h;

	if ((2 == nsegments) && (0 == ftcseg[0]) && (ftcseg[1] == nbins - 1))
	{
		nsegments = 1;
	}

	for (i = 0; i < nsegments; i++)
	{
		if (i < nsegments - 1)
		{
			for (k = ftcseg[i]; k < ftcseg[i + 1]; k++)
				idSegments[k] = i;
		}
		else
		{
			for (k = ftcseg[i]; k < nbins; k++)
				idSegments[k] = i;
			for (k = 0; k < ftcseg[0]; k++)
				idSegments[k] = i;
		}
	}

	for (i = 0; i < nsegments; i++)
	{
		Hsegment Hseg{};
		Hseg.Hmin = ftcseg[i] * qH;
		Hseg.Hmax = (i < nsegments - 1) ? (ftcseg[i + 1] * qH) : (ftcseg[0] * qH);
		hSegments.push_back(Hseg);
	}

	//assign each pixel to one of the segments
	for (n = 0; n < imgSize; n++)
	{
		iH = static_cast<int32_t>(hsi[n].H / qH);
		if (iH >= nbins)
			iH = nbins - 1;

		const int32_t& idseg = idSegments[iH];
		hSegments[idseg].pixels.push_back(n);
		hSegments[idseg].R += static_cast<float>(bgra[n].R);
		hSegments[idseg].G += static_cast<float>(bgra[n].G);
		hSegments[idseg].B += static_cast<float>(bgra[n].B);
	}

	//Get average RGB values for each segment
	for (i = 0; i < nsegments; i++)
	{
		const float& pSize = static_cast<float>(hSegments[i].pixels.size());
		hSegments[i].R = hSegments[i].R / pSize + 0.5f;
		hSegments[i].G = hSegments[i].G / pSize + 0.5f;
		hSegments[i].B = hSegments[i].B / pSize + 0.5f;
	}

	return hSegments;
}


template <typename T>
inline void channel_segmentation_saturation
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const T* __restrict bgra,
	struct Hsegment& hSeg,
	int32_t nbins,
	float q,
	std::vector<int32_t>& ftcseg
) noexcept
{
	CACHE_ALIGN int32_t idSegment[256]{};
	const int32_t& nsegments = static_cast<int32_t>(ftcseg.size()) - 1;
	int32_t ii, k;
	int32_t i;

	//assign same ID to all bins belonging to same segment
	for (ii = 0; ii < nsegments; ii++)
	{
		for (k = ftcseg[ii]; k <= ftcseg[ii + 1]; k++)
			idSegment[k] = ii;
	}

	//store information of each segment in a data structure (for each Hsegment)
	for (ii = 0; ii < nsegments; ii++)
	{
		Ssegment sSeg {};
		sSeg.Smin = ftcseg[ii] * q;
		sSeg.Smax = ftcseg[ii + 1] * q;
		hSeg.sSegments.push_back(sSeg);
	}

	//assign each pixel to one of the segments
	const int32_t& pSize = static_cast<int32_t>(hSeg.pixels.size());

	for (k = 0; k < pSize; k++)
	{
		const int32_t& n = hSeg.pixels[k];
		i = static_cast<int32_t>(hsi[n].S / q);
		
		if (i >= nbins)
			i = nbins - 1;

		const int32_t& idSeg = idSegment[i];
		hSeg.sSegments[idSeg].pixels.push_back(n);
		hSeg.sSegments[idSeg].R += static_cast<float>(bgra[n].R);
		hSeg.sSegments[idSeg].G += static_cast<float>(bgra[n].G);
		hSeg.sSegments[idSeg].B += static_cast<float>(bgra[n].B);
	}

	//Get average RGB values for each segment
	const int32_t& vSize = static_cast<int32_t>(hSeg.sSegments.size());
	for (i = 0; i < vSize; i++)
	{
		const int32_t& pSize = static_cast<int32_t>(hSeg.sSegments[i].pixels.size());
		hSeg.sSegments[i].R = hSeg.sSegments[i].R / pSize + 0.5f;
		hSeg.sSegments[i].G = hSeg.sSegments[i].G / pSize + 0.5f;
		hSeg.sSegments[i].B = hSeg.sSegments[i].B / pSize + 0.5f;
	}
	return;
}



std::vector<Hsegment> compute_color_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const PF_Pixel_BGRA_8u* __restrict bgra,
	float Smin,
	int32_t nbins,
	int32_t nbinsS,
	int32_t nbinsI,
	float qH,
	float qS,
	float qI,
	std::vector<int32_t>& ftcseg,
	int32_t w,
	int32_t h,
	float eps
) noexcept
{
	std::vector<Hsegment>&& hSegments = std::move(hue_segmentation (hsi, bgra, Smin, nbins, qH, ftcseg, w, h));
	const int32_t nsegmentsH = static_cast<int32_t>(hSegments.size());
	int32_t i, j, k, iS = 0;

	for (i = 0; i < nsegmentsH; i++)
	{
		CACHE_ALIGN int32_t histS[256]{};

		const int32_t pSize = static_cast<int32_t>(hSegments[i].pixels.size());
		for (k = 0; k < pSize; k++)
		{
			iS = static_cast<int32_t>(hsi[hSegments[i].pixels[k]].S / qS);
			if (iS >= nbinsS)
				iS = nbinsS - 1;
			histS[iS]++;
		}
		//segment saturation histogram
		std::vector<int>&& ftcsegS = std::move(ftc_utils_segmentation(histS, nbinsS, eps, false));

		channel_segmentation_saturation(hsi, bgra, hSegments[i], nbinsS, qS, ftcsegS);
		const int32_t& nsegmentsS = static_cast<int32_t>(hSegments[i].sSegments.size());

		for (j = 0; j < nsegmentsS; j++)
		{
			CACHE_ALIGN int32_t histI[256]{};

			const int32_t& pSizeS = static_cast<int32_t>(hSegments[i].sSegments[j].pixels.size());
			for (k = 0; k < pSizeS; k++)
			{
				int32_t iI = static_cast<int32_t>(hsi[hSegments[i].sSegments[j].pixels[k]].I / qI);
				if (iI >= nbinsI)
					iI = nbinsI - 1;
				histI[iI]++;
			}

			//segment histogram
			std::vector<int32_t>&& ftcsegI = std::move(ftc_utils_segmentation(histI, nbinsI, eps, false));

																		  //Obtain list of pixels associated to each mode of the Saturation histogram
//			channel_segmentation_intensity(I, R, G, B, Hsegments[i].Ssegments[j], nbinsI, qI, ftcsegI);
		}

	}

	return hSegments;
}