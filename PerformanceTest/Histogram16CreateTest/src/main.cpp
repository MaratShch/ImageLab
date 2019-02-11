#include <windows.h>
#include <stdio.h>
#include <time.h>

// include compiler intrinsict
#include <intrin.h>

#define DATA_SIZE_X		1920
#define DATA_SIZE_Y		1080
#define DATA_LINE_PITCH	1984
#define MAX_TEST_LOOP	16
#define HIST_POOL_SIZE	4
#define RANDOM_DATA_LIMIT 1024

typedef struct {
//LARGE_INTEGER totalIterClcks;
LARGE_INTEGER iterClcks[MAX_TEST_LOOP];
} trivialStatistics;

#if 0
unsigned short randomData; alignas(64)[DATA_SIZE_Y][DATA_LINE_PITCH];
int histogramData alignas(64)[65536];
#else
unsigned short* randomData;
int* histogramData;
#endif

trivialStatistics tStat1;
trivialStatistics tStat2;

__declspec(noinline) void allocateHistogramPull(int** pHistogram)
{
	const int totalPools = (HIST_POOL_SIZE)+1;
	// create interleaved histogram buffer
	pHistogram[0] = reinterpret_cast<int*>(malloc(65536 * sizeof(int) * (HIST_POOL_SIZE)));
	// create final histogram buffer
	pHistogram[1] = reinterpret_cast<int*>(malloc(65536 * sizeof(int)));
	return;
}

__declspec(noinline) void freeHistogramPull(int** pHistogram)
{
	if (nullptr != pHistogram[0])
	{
		free(pHistogram[0]);
		pHistogram[0] = nullptr;
	}
	if (nullptr != pHistogram[1])
	{
		free(pHistogram[1]);
		pHistogram[1] = nullptr;
	}

	return;
}


__declspec(noinline) void createTrivialHistogram(const unsigned short* __restrict pData, const int sizeX, const int sizeY, const int linePitch, int* __restrict pHistogram)
{
	unsigned short* pImage = const_cast<unsigned short*>(pData);
	unsigned short* pLine;

	for (int i = 0; i < sizeY; i++)
	{
		pLine = (unsigned short*)((__int64)(pData) + i * linePitch * sizeof(unsigned short));

		for (int j = 0; j < sizeX; j++)
		{
			pHistogram[pLine[j]] ++;
		}
	}
}

// memory optimized histogram computation
__declspec(noinline) void createOptimize1dHistogram(const unsigned short* pData, const int sizeX, const int sizeY, const int linePitch, int** pHistogram)
{
#if 0
	const unsigned short* __restrict pImage = pData;
	int* __restrict pHistogramInterleaved = pHistogram[0];
	int* __restrict pHistogramFinal = pHistogram[1];

	int* __restrict hist0 = &pHistogramInterleaved[0];
	int* __restrict hist1 = &pHistogramInterleaved[1];
	int* __restrict hist2 = &pHistogramInterleaved[2];
	int* __restrict hist3 = &pHistogramInterleaved[3];


	int    i, j;
	unsigned int    ha0o, bn0e, hs0e;
	unsigned int    ha0e, bn0o, hs0o;
	unsigned int    ha1o, bn1e, hs1e;
	unsigned int    ha1e, bn1o, hs1o;
	unsigned int    ha2o, bn2e, hs2e;
	unsigned int    ha2e, bn2o, hs2o;
	unsigned int    ha3o, bn3e, hs3e;
	unsigned int    ha3e, bn3o, hs3o;

	/* ---------------------------------------------------------------- */
	/*  Seed 'previous odd pixel' with bogus bin values that won't      */
	/*  match the first even pixels.                                    */
	/* ---------------------------------------------------------------- */

	bn3o = 1;
	bn2o = 1;
	bn1o = 1;
	bn0o = 1;

	/* ---------------------------------------------------------------- */
	/*  Prefetch the initial bins for each of our four histograms.      */
	/*  This is needed for properly handling our forwarding logic.      */
	/* ---------------------------------------------------------------- */
	ha3o = hist3[bn3o];    /* read previous odd data */
	ha2o = hist2[bn2o];    /* read previous odd data */
	ha1o = hist1[bn1o];    /* read previous odd data */
	ha0o = hist0[bn0o];    /* read previous odd data */

						   /* make interleaved histogram */
	for (j = 0; j < sizeY; j++)
	{
		const unsigned short* __restrict pLine = (unsigned short*)((__int64)(pData) + j * linePitch * sizeof(unsigned short));

		/* make interleaved histogram from single line */
		for (i = 0; i < sizeX; i += 8)
		{
			/*  Load 4 pixels from the even side of the image. */
			bn0e = pLine[0];
			bn1e = pLine[1];
			bn2e = pLine[2];
			bn3e = pLine[3];

			hs3e = hist3[bn3e];      /* Get even bin.                */
			hs2e = hist2[bn2e];      /* Get even bin.                */
			hs1e = hist1[bn1e];      /* Get even bin.                */
			hs0e = hist0[bn0e];      /* Get even bin.                */

			hist3[bn3o] = ha3o;      /* Save previous odd bin.       */
			hist2[bn2o] = ha2o;      /* Save previous odd bin.       */
			hist1[bn1o] = ha1o;      /* Save previous odd bin.       */
			hist0[bn0o] = ha0o;      /* Save previous odd bin.       */

			ha3e = 1 + hs3e;         /* Update even bin.             */
			ha2e = 1 + hs2e;         /* Update even bin.             */
			ha1e = 1 + hs1e;         /* Update even bin.             */
			ha0e = 1 + hs0e;         /* Update even bin.             */

			ha3e += (bn3e == bn3o);   /* Add forwarding.              */
			ha2e += (bn2e == bn2o);   /* Add forwarding.              */
			ha1e += (bn1e == bn1o);   /* Add forwarding.              */
			ha0e += (bn0e == bn0o);   /* Add forwarding.              */

		  /* ------------------------------------------------------------ */
		  /*  Load 4 pixels from the odd side of the image.               */
		  /* ------------------------------------------------------------ */
			bn0o = pLine[4];
			bn1o = pLine[5];
			bn2o = pLine[6];
			bn3o = pLine[7];

			hs3o = hist3[bn3o];      /* Get odd bin.                 */
			hs2o = hist2[bn2o];      /* Get odd bin.                 */
			hs1o = hist1[bn1o];      /* Get odd bin.                 */
			hs0o = hist0[bn0o];      /* Get odd bin.                 */

			hist3[bn3e] = ha3e;      /* Save previous even bin.      */
			hist2[bn2e] = ha2e;      /* Save previous even bin.      */
			hist1[bn1e] = ha1e;      /* Save previous even bin.      */
			hist0[bn0e] = ha0e;      /* Save previous even bin.      */

			ha3o = 1 + hs3o;         /* Update odd bin.              */
			ha2o = 1 + hs2o;         /* Update odd bin.              */
			ha1o = 1 + hs1o;         /* Update odd bin.              */
			ha0o = 1 + hs0o;         /* Update odd bin.              */

			ha3o += (bn3o == bn3e);   /* Add forwarding.              */
			ha2o += (bn2o == bn2e);   /* Add forwarding.              */
			ha1o += (bn1o == bn1e);   /* Add forwarding.              */
			ha0o += (bn0o == bn0e);   /* Add forwarding.              */

		}
	}

	/* ---------------------------------------------------------------- */
	/*  Store final odd-pixel bin values.                               */
	/* ---------------------------------------------------------------- */
	hist3[bn3o] = ha3o;
	hist2[bn2o] = ha2o;
	hist1[bn1o] = ha1o;
	hist0[bn0o] = ha0o;


	/* histogram finalization - make final histogram from interleaved subs...*/
	for (j = 0; j < 65536; j++)
	{
		pHistogramFinal[j] = hist0[j] + hist1[j] + hist2[j] + hist3[j];
	}
#endif

	return;
}


__declspec(noinline) void generate_random_data(unsigned short* pData, const int sizeY, const int linePitch)
{
	const int poolSize = sizeY * linePitch;
	srand((unsigned)time(NULL));

	for (int i = 0; i < poolSize; i++)
	{
		pData[i] = (unsigned short)(rand()) % (RANDOM_DATA_LIMIT);
	}
}


int main(void)
{
	void* pMemPool;
	int* pHistogram[(HIST_POOL_SIZE)+1] = { nullptr };

	LARGE_INTEGER tS1, tS2, tF;
	__int64 tsMin1, tsMax1;
	__int64 tsMin2, tsMax2;

	int loopCnt = 0;

	tF.QuadPart = 0ll;
	QueryPerformanceFrequency(&tF);

	memset(&tStat1, 0, sizeof(tStat1));
	memset(&tStat2, 0, sizeof(tStat2));

	allocateHistogramPull(pHistogram);

	randomData = nullptr;
	histogramData = nullptr;

	pMemPool = nullptr;
	const size_t dataPoolSize = (DATA_SIZE_Y) * (DATA_LINE_PITCH) * sizeof(unsigned short);
	pMemPool = malloc(dataPoolSize);
	if (nullptr != pMemPool)
	{
		memset(pMemPool, 0, dataPoolSize);
		randomData = reinterpret_cast<unsigned short*>(pMemPool);
	}

	pMemPool = nullptr;
	const size_t histPoolSize = 65536 * sizeof(int);
	pMemPool = malloc(histPoolSize);
	if (nullptr != pMemPool)
	{
		memset(pMemPool, 0, histPoolSize);
		histogramData = reinterpret_cast<int*>(pMemPool);
	}

	// test loop entry point 
	if (nullptr != histogramData && nullptr != randomData)
	{

		for (loopCnt = 0; loopCnt < (MAX_TEST_LOOP); loopCnt++)
		{
			// create random data
			generate_random_data((unsigned short*)randomData, DATA_SIZE_Y, DATA_LINE_PITCH);

			// cleanup histogram buffer
			memset(reinterpret_cast<void*>(histogramData), 0, histPoolSize);

			QueryPerformanceCounter(&tS1);
			// create trivial histogram
			createTrivialHistogram((unsigned short*)randomData, DATA_SIZE_X, DATA_SIZE_Y, DATA_LINE_PITCH, histogramData);
			QueryPerformanceCounter(&tS2);
			tStat1.iterClcks[loopCnt].QuadPart = tS2.QuadPart - tS1.QuadPart;

			QueryPerformanceCounter(&tS1);
			// create optimized1 histogram
			createOptimize1dHistogram((unsigned short*)randomData, DATA_SIZE_X, DATA_SIZE_Y, DATA_LINE_PITCH, pHistogram);
			QueryPerformanceCounter(&tS2);
			tStat2.iterClcks[loopCnt].QuadPart = tS2.QuadPart - tS1.QuadPart;

		}

		// cleanup memory
		free(randomData);
		randomData = nullptr;
		free(histogramData);
		histogramData = nullptr;

	}

	freeHistogramPull(pHistogram);

	printf("Run-time complete. Analyze statistics...\n");

	tsMin1 = tsMin2 = MAXINT64;
	tsMax1 = tsMax2 = MININT64;
	for (loopCnt = 0; loopCnt < (MAX_TEST_LOOP); loopCnt++)
	{
		if (tStat1.iterClcks[loopCnt].QuadPart < tsMin1)
			tsMin1 = tStat1.iterClcks[loopCnt].QuadPart;
		if (tStat1.iterClcks[loopCnt].QuadPart > tsMax1)
			tsMax1 = tStat1.iterClcks[loopCnt].QuadPart;

		if (tStat2.iterClcks[loopCnt].QuadPart < tsMin2)
			tsMin2 = tStat2.iterClcks[loopCnt].QuadPart;
		if (tStat2.iterClcks[loopCnt].QuadPart > tsMax2)
			tsMax2 = tStat2.iterClcks[loopCnt].QuadPart;
	}

	printf("Trivial Statistics [CLCK RESOLUTION = %I64d]:", tF.QuadPart);
	printf("Number of iteration: %d\n", MAX_TEST_LOOP);
	for (loopCnt = 0; loopCnt < (MAX_TEST_LOOP); loopCnt++)
	{
		printf("[%03d] T: %I64d  O1: %I64d\n", loopCnt, tStat1.iterClcks[loopCnt].QuadPart, tStat2.iterClcks[loopCnt].QuadPart);
	}
	printf("MIN/MAX values: T : MIN = %I64d\t MAX = %I64d\n", tsMin1, tsMax1);
	printf("MIN/MAX values: O1: MIN = %I64d\t MAX = %I64d\n", tsMin2, tsMax2);

	printf("Complete. Press <ENTER> for exit...\n");
	(void)getchar();


	return EXIT_SUCCESS;
}