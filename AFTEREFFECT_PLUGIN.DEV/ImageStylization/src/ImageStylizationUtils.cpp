#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
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

void utils_create_random_buffer(void)
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> get_minima (int32_t* hist, int size)
{
	CACHE_ALIGN int type[hist_size_H * circular_size]{};
	std::vector<int> minpos;
	
	//type of each point (bin) in histogram: 
	//1: minimum (value < previous, value < next)
	//2: potential left endpoint of flat minimum (value < previous, value = next)
	//3: potential right endpoint of flat minimum (value < next, value = previous)
	//4: flat point (value=previous=next)
	//5: endpoint of histogram
	//0: rest of cases
	float diffprev, diffnext;
	type[0] = 5;
	type[size - 1] = 5;
	for (int i = 1; i < size - 1; i++) {
		type[i] = 0;
		diffprev = hist[i] - hist[i - 1];
		diffnext = hist[i] - hist[i + 1];
		if ((diffprev < 0) && (diffnext < 0)) type[i] = 1; //minimum
		if ((diffprev == 0) && (diffnext == 0)) type[i] = 4; //flat
		if ((diffprev < 0) && (diffnext == 0)) type[i] = 2; //potential left endpoint of flat minimum
		if ((diffprev == 0) && (diffnext < 0)) type[i] = 3; //potential right endpoint of flat minimum 
	}
	//check flat minima
	for (int i = 1; i < size - 1; i++) {
		if (type[i] == 2) { //potential left endpoint of flat minimum
							//look for right endpoint
			int j;
			for (j = i + 1; (j < size - 1) && (type[j] == 4); j++);
			if (type[j] == 3) { //found right endpoint
								//mark center of flat zone as minimum
				type[(i + j) / 2] = 1;
			}
		}
	}

	//output list of minima + endpoints
	minpos.push_back(0); //left endpoint
	for (int i = 1; i < size - 1; i++) {
		if (type[i] == 1) minpos.push_back(i); //minimum
	}
	minpos.push_back(size - 1); //right endpoint
	return minpos;
}


//get list of maxima of a histogram
std::vector<int> get_maxima (int32_t* hist, int size)
{
	CACHE_ALIGN int type[hist_size_H * circular_size]{};
	std::vector<int> maxpos;

	//type of each point (bin) in histogram: 
	//1: maximum (value > previous, value > next)
	//2: potential left endpoint of flat maximum (value > previous, value = next)
	//3: potential right endpoint of flat maximum (value > next, value = previous)
	//4: flat point (value=previous=next)
	//0: rest of cases
	float diffprev, diffnext;
	//check all except endpoints
	for (int i = 1; i < size - 1; i++) {
		type[i] = 0;
		diffprev = hist[i] - hist[i - 1];
		diffnext = hist[i] - hist[i + 1];
		if ((diffprev > 0) && (diffnext > 0)) type[i] = 1; //maximum
		if ((diffprev == 0) && (diffnext == 0)) type[i] = 4; //flat
		if ((diffprev > 0) && (diffnext == 0)) type[i] = 2; //potential left endpoint of flat maximum
		if ((diffprev == 0) && (diffnext > 0)) type[i] = 3; //potential right endpoint of flat maximum 
	}
	//check endpoints
	type[0] = 0;
	type[size - 1] = 0;
	if (hist[0] > hist[1]) type[0] = 1; //maximum
	if (hist[0] == hist[1]) type[0] = 2; //potential left endpoint of flat maximum
	if (hist[size - 1] > hist[size - 2]) type[size - 1] = 1; //maximum
	if (hist[size - 1] == hist[size - 2]) type[size - 1] = 3; //potential right endpoint of flat maximum 

															  //check flat maximum
	for (int i = 0; i < size; i++) {
		if (type[i] == 2) { //potential left endpoint of flat maximum
							//look for right endpoint
			int j;
			for (j = i + 1; (j < size - 1) && (type[j] == 4); j++);
			if (type[j] == 3) { //found right endpoint
								//mark center of flat zone as maximum
				type[(i + j) / 2] = 1;
			}
		}
	}

	//output list of maxima 
	for (int i = 0; i < size; i++) {
		if (type[i] == 1) maxpos.push_back(i); //maximum
	}

	return maxpos;
}


//check if the cost of merging two intervals has been already computed
unsigned char cost_already_computed(std::vector<costdata> &listcosts,
	int imin1, int imin2, struct costdata &cdata)
{
	unsigned char found = 0;
	for (int k = 0; k < listcosts.size() && !found; k++) {
		if ((listcosts[k].imin1 == imin1) && (listcosts[k].imin2 == imin2)) {
			found = 1;
			cdata = listcosts[k];
		}
	}

	return found;
}

//find monotonically decreasing and increasing intervals of the histogram
//type of each point (bin) in histogram: 
//0: flat (current = previous)
//1: increasing (current > previous)
//2: decreasing (current < previous)
//This function implements lines 3-6 of the Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
void get_monotone_info(float *hist, int size, int *type,
	int &nincreasing, int &ndecreasing,
	unsigned char extend_increasing)
{
	nincreasing = 0;
	ndecreasing = 0;

	float diffprev;
	//assume first bin = flat
	type[0] = 0;
	for (int i = 1; i < size; i++) {
		diffprev = hist[i] - hist[i - 1];
		type[i] = 0;
		if (diffprev > 0) {
			type[i] = 1;
			nincreasing++;
		}
		else {
			if (diffprev < 0) {
				type[i] = 2;
				ndecreasing++;
			}
		}
	}

	//extend strict monotony type (< or >) to general monotony (<= or >=)
	int typeV;
	if (extend_increasing) typeV = 1;
	else typeV = 2;

	//extend to the left of non-flat bin
	int ilast = -1;
	for (int i = 1; i < size; i++) {
		if (type[i] == typeV) { //non-flat bin
			for (int j = i - 1; (j > 0) && (type[j] == 0); j--) type[j] = typeV;
			ilast = i;
		}
	}

	//last non-flat bin: extend to the right
	if (ilast != -1) for (int j = ilast + 1; (j < size) && (type[j] == 0); j++) type[j] = typeV;


}

//replace a monotonically increasing (resp. decreasing) interval of the 
//histogram by a constant value
//This code implements lines 7-9 of the Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
void replace_monotone(float *hist, int size, int *type,
	unsigned char replace_increasing)
{
	int typeV;
	if (replace_increasing) typeV = 1;
	else typeV = 2;

	//find monotonically decreasing/increasing intervals
	int i = 0;
	int istart, iend;
	unsigned char isfirst = 1; //flag that indicates beginning of interval
	while (i < size) {
		if (type[i] == typeV) {
			if (isfirst) {
				//istart=i;
				istart = (i > 0) ? (i - 1) : i; //take value to the left of left endpoint of interval
				isfirst = 0;
			}
			if ((i == size - 1) || (type[i + 1] != typeV)) {
				iend = i;
				//assign constant value to interval
				float C = 0;
				for (int j = istart; j <= iend; j++) C += hist[j];
				C /= (iend - istart + 1);
				for (int j = istart; j <= iend; j++) hist[j] = C;
				isfirst = 1;
			}
		}
		i++;
	}

}

//Implement Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
float *pool_adjacent_violators(float *hist, int size, unsigned char increasing)
{
	float *hmono = new float[size];
	//initialize with input histogram
	memcpy(hmono, hist, size * sizeof(float));

	//type of each point (bin) in histogram: 
	//0: flat (current = previous)
	//1: increasing (current > previous)
	//2: decreasing (current < previous)
	int *type = new int[size];
	int nincreasing, ndecreasing;

	if (increasing) { //get increasing histogram
		do {
			//find monotonically decreasing intervals of the histogram
			get_monotone_info(hmono, size, type, nincreasing, ndecreasing, 0);
			//replace monotonically decreasing intervals by a constant value 
			if (ndecreasing > 0) {
				replace_monotone(hmono, size, type, 0);
			}
		} while (ndecreasing > 0);//stop when no more decreasing intervals exist    
	}
	else {    //get decreasing histogram
		do {
			//find monotonically increasing intervals of the histogram
			get_monotone_info(hmono, size, type, nincreasing, ndecreasing, 1);
			//replace monotonically decreasing intervals by a constant value 
			if (nincreasing > 0) {
				replace_monotone(hmono, size, type, 1);
			}
		} while (nincreasing > 0);//stop when no more increasing intervals exist   
	}

	delete[] type;
	return hmono;
}

//compute relative entropy value
double relative_entropy(double r, double p)
{
	double H;
	if (r == 0.0) H = -log(1.0 - p);
	else if (r == 1.0) H = -log(p);
	else H = (r*log(r / p) + (1.0 - r)*log((1.0 - r) / (1.0 - p)));

	return H;
}

double cost_monotone(int32_t *hist0, int i1, int i2, unsigned char increasing, double logeps)
{
	//double logeps=0; //logeps=log(NFAmax)  
	//NFAmax=maximum expected number of false positives w.r.t. null hypothesis (monotony)

	//get subhistogram
	int L = i2 - i1 + 1;
	float *hist = new float[L];
	for (int i = 0; i < L; i++) hist[i] = hist0[i1 + i];

	//get monotone estimation
	float *hmono = pool_adjacent_violators(hist, L, increasing);

	//Compute cost

	//cumulated histograms
	for (int i = 1; i < L; i++) hist[i] += hist[i - 1];
	for (int i = 1; i < L; i++) hmono[i] += hmono[i - 1];
	//meaningfullness threshold
	int N = hist[L - 1];
	double threshold = (log((double)L*(L + 1) / 2) - logeps) / (double)N;

	//find interval that more rejects the null hypothesis (monotony)
	//i.e. highest Kullblack-Leibler distance from hist to hmono
	double r, p, H, Hmax;
	for (int i = 0; i < L; i++)
		for (int j = i; j < L; j++) {
			//r: proportion of values in [i, j], for hist 
			if (i == 0) r = (double)hist[j];
			else r = (double)(hist[j] - hist[i - 1]);
			r = r / (double)N;
			//p: proportion of values in [i, j], for hmono            
			if (i == 0) p = (double)hmono[j];
			else p = (double)(hmono[j] - hmono[i - 1]);
			p = p / (double)N;
			//relative entropy (Kullblack-Leibler distance from hist to hmono)
			H = relative_entropy(r, p);
			if (((i == 0) && (j == 0)) || (H > Hmax)) Hmax = H;

		}

	//cost
	double cost = (double)N * Hmax - (log((double)L*(L + 1) / 2) - logeps);

	//    printf("        Hmax=%e   N=%i  L=%i  cost=%e\n", Hmax, N, L, cost);

	delete[] hist;
	delete[] hmono;
	hmono = hist = nullptr;

	return cost;
}

//each mode is composed of:
// minimum - maximum - minimum
// we want to check the if the unimodal hypothesis is preserved by merging
// minimumA - maximumA - ... - maximumB - minimumB
// maximumA is the maximum to the right of minimumA
// maximumB is the maximum to the left of minimumB
// i1 is the index of minimumA in separators list
// i2 is the index of minimumB in separators list
// i1 is the index of maximumA in maxima list
// i2-1 is the index of maximumB in maxima list
// the intervals can be merged if 
// 1) minimumA-maximumB is monotonically increasing (typemerging = 1)
// or
// 2) maximumA-minimumB is monotonically decreasing (typemerging = 2)
// If 1) then minimumA-maximumB-minimumB is unimodal 
//       and maximumA-... can be removed from the list of minima/maxima
// If 2) then minimumA-maximumA-minimumB is unimodal 
//       and  ...-maximumB can be removed from the list of minima/maxima
struct costdata cost_merging(int32_t *hist, std::vector<costdata> &listcosts,
	std::vector<int> &separators, std::vector<int> &maxima,
	int i1, int i2, double logeps)
{
	struct costdata cdata;

	//maximumB == maximum at position imin2-1
	double cost1 = cost_monotone(hist, separators[i1], maxima[i2 - 1], 1, logeps); //increasing
	double cost2 = cost_monotone(hist, maxima[i1], separators[i2], 0, logeps); //decreasing
	double cost;
	//keep the smallest
	if (cost1 < cost2) {
		cdata.cost = cost1;
		cdata.typemerging = 1;
	}
	else {
		cdata.cost = cost2;
		cdata.typemerging = 2;
	}
	cdata.imin1 = separators[i1];
	cdata.imin2 = separators[i2];

	listcosts.push_back(cdata);

	return cdata;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t ftc_utils_get_minima (const int32_t* __restrict pHist, int32_t* __restrict vectorOut, const int32_t histSize) noexcept
{
	CACHE_ALIGN int type[256 /*hist_size_H * circular_size*/]{};

	const int32_t& lastSample = histSize - 1;
	float prev, curr, next;
	int32_t i, j, out_size;

	type[0] = 5;
	__VECTOR_ALIGNED__
	prev = pHist[0];
	for (i = 1; i < lastSample; i++)
	{
		curr = pHist[i    ];
		next = pHist[i + 1];

		type[i] = 0;
		const float& diffprev = curr - prev;
		const float& diffnext = curr - next;

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

	//output list of minima + endpoints
	out_size = 0;
	vectorOut[out_size] = 0; //left endpoint
	out_size++;

	for (i = 1; i < lastSample; i++) {
		if (1 == type[i])
		{
			vectorOut[out_size] = i; //minimum
			out_size++;
		}
	}
	vectorOut[out_size] = lastSample; //right endpoint

	return out_size;
}


inline int32_t ftc_utils_get_maxima (const int32_t* __restrict pHist, int32_t* __restrict vectorOut, const int32_t histSize) noexcept
{
	CACHE_ALIGN int type[256 /*hist_size_H * circular_size*/]{};

	const int32_t& lastSample = histSize - 1;
	float prev, curr, next;
	int32_t i, j, out_size;

	type[0] = 0;
	__VECTOR_ALIGNED__
	prev = pHist[0];
	for (i = 1; i < lastSample; i++)
	{
		curr = pHist[i    ];
		next = pHist[i + 1];

		type[i] = 0;
		const float& diffprev = curr - prev;
		const float& diffnext = curr - next;
	
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
	out_size = 0;
	for (i = 0; i < histSize; i++)
	{
		if (type[i] == 1)
		{
			/* maximum */
			vectorOut[out_size] = i;
			out_size++;
		}
	}

	return out_size;
}


void ftc_utils_segmentation(const int32_t* inHist, const int32_t& inHistSize, float epsilon, bool isGray) noexcept
{
	CACHE_ALIGN int32_t circularH [hist_size_H * circular_size];
	int32_t* pHistogram = nullptr;
	const float fLogEps = log(epsilon);
	int32_t histSize;
	const bool& circularHist = !isGray;

	/* make circular histogram */
	if (false == isGray)
	{
		memset(circularH, 0, sizeof(circularH));
		__VECTOR_ALIGNED__
		for (int i = 0; i < inHistSize; i++)
		{
			circularH[i                 ] = inHist[i];
			circularH[i + inHistSize    ] = inHist[i];
			circularH[i + 2 * inHistSize] = inHist[i];
		}
		pHistogram = circularH;
		histSize = inHistSize * circular_size;
	}
	else
	{
		pHistogram = const_cast<int32_t*>(inHist);
		histSize = inHistSize;
	}

	CACHE_ALIGN int32_t pMinimaVectorOut[256]{};
	const int32_t& minimaSize = ftc_utils_get_minima (pHistogram, pMinimaVectorOut, histSize);

	CACHE_ALIGN int32_t pMaximaVectorOut[256]{};
	const int32_t& maximaSize = ftc_utils_get_maxima (pHistogram, pMaximaVectorOut, histSize);


	return;
}