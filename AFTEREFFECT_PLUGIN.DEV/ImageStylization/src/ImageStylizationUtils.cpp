#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "FastAriphmetics.hpp"
#include <vector>
#include <array>

CACHE_ALIGN static float RandomValBuffer[RandomBufSize]{};

//data structure to store the cost of merging intervals of the histogram
struct costdata {
	double cost;
	int imin1, imin2;
	int typemerging;
};

uint32_t utils_get_random_value(void) noexcept
{
	// used xorshift algorithm
	static uint32_t x = 123456789u;
	static uint32_t y = 362436069u;
	static uint32_t z = 521288629u;
	static uint32_t w = 88675123u;
	static uint32_t t;

	t = x ^ (x << 11);
	x = y; y = z; z = w;
	return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}


void utils_generate_random_values (float* pBuffer, const uint32_t& bufSize) noexcept
{
	constexpr float fLimit = static_cast<float>(UINT_MAX);
	if (nullptr != pBuffer && 0u != bufSize)
	{
		for (uint32_t idx = 0u; idx < bufSize; idx++)
		{
			pBuffer[idx] = static_cast<float>(utils_get_random_value()) / fLimit;
		}
	}
	return;
}

void utils_create_random_buffer(void) noexcept
{
	static_assert(IsPowerOf2(RandomBufSize), "Random buffer size isn't power of 2");
	utils_generate_random_values(RandomValBuffer, RandomBufSize);
	return;
}


const float* __restrict get_random_buffer (uint32_t& size) noexcept
{
	size = RandomBufSize;
	return RandomValBuffer;
}

const float* __restrict get_random_buffer (void) noexcept 
{
	return RandomValBuffer;
}


inline std::vector<int> ftc_utils_get_minima (const int32_t* __restrict pHist, const int32_t histSize) noexcept
{
	CACHE_ALIGN int type[256 /*hist_size_H * circular_size*/]{};
	std::vector<int> vectorOut{};

	const int32_t& lastSample = histSize - 1;
	int32_t prev, curr, next;
	int32_t i, j;

	type[0] = 5;
	__VECTOR_ALIGNED__
	prev = pHist[0];
	for (i = 1; i < lastSample; i++)
	{
		curr = pHist[i    ];
		next = pHist[i + 1];

		type[i] = 0;
		const int32_t& diffprev = curr - prev;
		const int32_t& diffnext = curr - next;

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
			vectorOut.push_back( i ); //minimum
		}
	}
	vectorOut.push_back(lastSample); //right endpoint
	return vectorOut;
}


inline std::vector<int> ftc_utils_get_maxima (const int32_t* __restrict pHist, const int32_t histSize) noexcept
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
		curr = pHist[i    ];
		next = pHist[i + 1];

		type[i] = 0;
		const int32_t& diffprev = curr - prev;
		const int32_t& diffnext = curr - next;
	
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



inline bool cost_already_computed (std::vector<CostData>& listCosts, const int32_t& imin1, const int32_t& imin2, CostData& cdata) noexcept
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


void get_monotone_info (float* __restrict hist, int32_t* __restrict type, int32_t size, int &nincreasing, int &ndecreasing, bool extend_increasing) noexcept
{
	int32_t i = 0, j = 0;
	type[0] = nincreasing = ndecreasing = 0;

	for (i = 1; i < size; i++)
	{
		const float& diffprev = hist[i] - hist[i - 1];
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

	const int32_t& typeV = extend_increasing ? 1 : 2;
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


inline void replace_monotone (float* __restrict hist, int32_t size, int32_t* __restrict type, bool replace_increasing) noexcept
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


inline void pool_adjacent_violators (float* __restrict hist, float* __restrict hMono, int32_t size, bool increasing) noexcept
{
	CACHE_ALIGN int32_t Size[256]{};
	CACHE_ALIGN int32_t Type[256]{};
	int32_t nincreasing, ndecreasing;

	const size_t& memSize = size * sizeof(float);
	memcpy (hMono, hist, memSize);

	if (increasing)
	{ 
		do {
			get_monotone_info (hMono, Type, size, nincreasing, ndecreasing, false);
			if (ndecreasing > 0)
				replace_monotone (hMono, size, Type, false);
		} while (ndecreasing > 0);   
	}
	else
	{
		do {
			get_monotone_info (hMono, Type, size, nincreasing, ndecreasing, true);
			if (nincreasing > 0)
				replace_monotone (hMono, size, Type, true);
		} while (nincreasing > 0);
	}
	return;
}


inline float relative_entropy (const float& r, const float& p) noexcept
{
	float H;

	if (r == 0.f)
		H = -FastCompute::Log(1.f - p);
	else if (r == 1.f)
		H = -FastCompute::Log(p);
	else 
		H = (r * FastCompute::Log(r / p) + (1.f - r) * FastCompute::Log((1.f - r) / (1.f - p)));

	return H;
}


inline float cost_monotone (const int32_t* __restrict hist0, int32_t i1, int32_t i2, bool increasing, float logeps) noexcept
{
	CACHE_ALIGN float hist [256]{};
	CACHE_ALIGN float hMono[256]{};

	const int32_t& L = i2 - i1 + 1;
	int32_t i, j;

	__VECTOR_ALIGNED__
	for (i = 0; i < L; i++)
		hist[i] = static_cast<float>(hist0[i1 + i]);

	//get monotone estimation
	pool_adjacent_violators (hist, hMono, L, increasing);

	//cumulated histograms
	for (i = 1; i < L; i++)
		hist[i]  += hist[i - 1];

	for (i = 1; i < L; i++)
		hMono[i] += hMono[i - 1];

	const int32_t& N = static_cast<int32_t>(hist[L - 1]);
	const float& threshold = (FastCompute::Log(static_cast<float>(L) * (L + 1) / 2) - logeps) / static_cast<float>(N);

	float H = 0.f, Hmax = FLT_MIN;

	for (i = 0; i < L; i++)
	{
		for (j = i; j < L; j++)
		{
			const float& r = (0 == i ? hist[j]  : hist[j]  - hist [i - 1]) / static_cast<float>(N);
			const float& p = (0 == i ? hMono[j] : hMono[j] - hMono[i - 1]) / static_cast<float>(N);

			H = relative_entropy(r, p);

			if (((i == 0) && (j == 0)) || (H > Hmax))
				Hmax = H;
		}
	}

	float const& cost = static_cast<float>(N) * Hmax - (std::log(static_cast<float>(L*(L + 1) >> 1)) - logeps);

	return cost;
}


inline CostData cost_merging (
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

	const float& cost1 = cost_monotone (hist, separatorsI1, maximaI2, 1, logeps); //increasing
	const float& cost2 = cost_monotone (hist, separatorsI1, separatorsI2, 0, logeps); //decreasing

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


std::vector<int32_t> ftc_utils_segmentation (const int32_t* inHist, const int32_t& inHistSize, float epsilon, bool isGray) noexcept
{
	CACHE_ALIGN int32_t circularH [hist_size_H * circular_size];
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
			circularH[i                ] = inHist[i];
			circularH[i + inHistSize   ] = inHist[i];
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
	std::vector<int>&& SeparatorVector = ftc_utils_get_minima (pHistogram, histSize);
	std::vector<int>&& MaximaVectorOut = ftc_utils_get_maxima (pHistogram, histSize);
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
			const int32_t& nIntervalMinusJ = nIntervals - j;

			for (i = 0; i < nIntervalMinusJ; i++)
			{
				auto const& i_j_sum = i + j + 1;
				auto const& SeparatorJ = SeparatorVector[i_j_sum];
				auto const& SeparatorI = SeparatorVector[i];
				auto const& SeparatorDiffs = SeparatorJ - SeparatorI;

				if (circularHist && SeparatorDiffs > histSize)
					continue;

				if (false == (costComputed = cost_already_computed (listCosts, SeparatorI, SeparatorJ, cData)))
				{
					cData = cost_merging (pHistogram, listCosts, SeparatorVector, MaximaVectorOut, i, i_j_sum, fLogEps);
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
				//remove maxima associated to the removed minima
				if (1 == cDataLowest.typemerging)
				{
					MaximaVectorOut.erase(MaximaVectorOut.begin() + iLowest, MaximaVectorOut.begin() + iLowest + j);
				}
				if (2 == cDataLowest.typemerging)
				{
					MaximaVectorOut.erase(MaximaVectorOut.begin() + iLowest + 1, MaximaVectorOut.begin() + iLowest + j + 1);
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
		const int32_t& doubleHistSize = 2 * inHistSize;

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

	return (false == isGray ? separatorsC: SeparatorVector);
}