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
LARGE_INTEGER totalIterClcks;
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
	pHistogram[0] = reinterpret_cast<int*>(malloc(65536 * sizeof(int)));
	return;
}

__declspec(noinline) void freeHistogramPull(int** pHistogram)
{
	for (int i = 0; i < 2; i++)
	{
		if (nullptr != pHistogram[i])
		{
			free(pHistogram[i]);
			pHistogram[i] = nullptr;
		}
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
	unsigned short* __restrict pImage = const_cast<unsigned short*>(pData);
	int* __restrict pHistogramInterleaved = pHistogram  [0];
	int* __restrict pHistogramFinal = pHistogram  [1];

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