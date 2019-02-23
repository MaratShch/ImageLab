#include <windows.h>
#include <stdio.h>
#include <time.h>

// include compiler intrinsict
#include <intrin.h>

#define DATA_SIZE_X					1920
#define DATA_SIZE_Y					1080
#define DATA_LINE_PITCH				1984
#define MAX_TEST_LOOP				16
#define RANDOM_DATA_LIMIT			1024

#define CACHE_LINE  32
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define DEFAULT_MEMORY_ALIGNMENT	CACHE_LINE

typedef struct {
	LARGE_INTEGER iterClck;
	int           errCnt;
	int           cnt;
}TRIVIAL_STATISTICS;


CACHE_ALIGN static TRIVIAL_STATISTICS statVal[2][MAX_TEST_LOOP];
CACHE_ALIGN static double statRelations[MAX_TEST_LOOP];

void generate_random_data(unsigned short* pData, const int sizeY, const int linePitch)
{
	const int poolSize = sizeY * linePitch;
	srand((unsigned)time(NULL));

	for (int i = 0; i < poolSize; i++)
	{
		pData[i] = (unsigned short)(rand()) % (RANDOM_DATA_LIMIT);
	}
}


void createTrivialHistogram16(void* __restrict pData,
							  const int sizeX,
							  const int sizeY,
							  const int linePitch,
							  void* __restrict pHist)
{
	unsigned short* __restrict pImage = reinterpret_cast<unsigned short*>(pData);
	unsigned int*   __restrict pHistogram = reinterpret_cast<unsigned int*>(pHist);

	for (int i = 0; i < sizeY; i++)
	{
		const unsigned short* pLine = reinterpret_cast<unsigned short*>(pImage + i * linePitch);

		for (int j = 0; j < sizeX; j++)
		{
			pHistogram[pLine[j]] ++;
		}
	}
}


void createTrivialHistogram8(void* __restrict pData,
							 const int sizeX,
							 const int sizeY,
							 const int linePitch,
							 void* __restrict pHist)
{
	unsigned char* __restrict pImage = reinterpret_cast<unsigned char*>(pData);
	unsigned int*  __restrict pHistogram = reinterpret_cast<unsigned int*>(pHist);

	for (int i = 0; i < sizeY; i++)
	{
		const unsigned char* pLine = reinterpret_cast<unsigned char*>(pImage + i * linePitch);

		for (int j = 0; j < sizeX; j++)
		{
			pHistogram[pLine[j]] ++;
		}
	}
}


int compare_histograms_results(const int* __restrict pTrivial, const int* __restrict pOptimized, const int size)
{
	int errors = 0;
	for (int i = 0; i < size; i++)
	{
		if (pTrivial[i] != pOptimized[i])
		{
			errors++;
		}
	}

	return errors;
}


int main(void)
{
	int loopCnt;
	int i;
	const size_t histogram16BufSize = sizeof(int) * 65536;
	const size_t histogram8BufSize = sizeof(int) * 256;
	const size_t histogram16OptBufSize = sizeof(int) * 65536 * 4;
	const size_t histogram8OptBufSize = sizeof(int) * 256 * 4;
	const size_t image16BufSize = (DATA_LINE_PITCH)* (DATA_SIZE_Y)* sizeof(unsigned short);
	const size_t image8BufSize  = (DATA_LINE_PITCH)* (DATA_SIZE_Y)* sizeof(unsigned short);
	const size_t memoryAlignemnt = DEFAULT_MEMORY_ALIGNMENT;

	LARGE_INTEGER ts1 = { 0 };
	LARGE_INTEGER ts2 = { 0 };
	
	// alloctae memory resources for buffers
	void* pImageBuf16 = _aligned_malloc(image16BufSize, memoryAlignemnt);
	void* pImageBuf8  = _aligned_malloc(image8BufSize,  memoryAlignemnt);

	void* pTrivialHist16Buffer = _aligned_malloc(histogram16BufSize, memoryAlignemnt);
	void* pTrivialHist8Buffer  = _aligned_malloc(histogram8BufSize, memoryAlignemnt);

	printf("Allocate histogram buffer %p for 8  bits image %d bytes [aligned on %d bytes]\n", pTrivialHist8Buffer,  (int)histogram8BufSize,  DEFAULT_MEMORY_ALIGNMENT);
	printf("Allocate histogram buffer %p for 16 bits image %d bytes [aligned on %d bytes]\n", pTrivialHist16Buffer, (int)histogram16BufSize, DEFAULT_MEMORY_ALIGNMENT);
	printf("\n");

	void* pOptHist16Buffer = _aligned_malloc(histogram16OptBufSize, memoryAlignemnt);
	void* pOptHist8Buffer  = _aligned_malloc(histogram8OptBufSize,  memoryAlignemnt);
	void* pOptFinalHist16Buffer = _aligned_malloc(histogram16BufSize, memoryAlignemnt);
	void* pOptFinalHist8Buffer  = _aligned_malloc(histogram8BufSize, memoryAlignemnt);

	printf("Allocate opt_histogram buffer %p for 8  bits image %d bytes [aligned on %d bytes]\n", pOptHist8Buffer,  (int)histogram8OptBufSize,  DEFAULT_MEMORY_ALIGNMENT);
	printf("Allocate opt_histogram buffer %p for 16 bits image %d bytes [aligned on %d bytes]\n", pOptHist16Buffer, (int)histogram16OptBufSize, DEFAULT_MEMORY_ALIGNMENT);
	printf("\n");


	if (nullptr != pImageBuf16 && nullptr != pImageBuf8 && nullptr != pTrivialHist16Buffer && nullptr != pTrivialHist8Buffer
		&& nullptr != pOptHist16Buffer && nullptr != pOptHist8Buffer && nullptr != pOptFinalHist16Buffer && nullptr != pOptFinalHist8Buffer)
	{
		memset(&statVal, 0, sizeof(statVal));
		memset(&statRelations, 0, sizeof(statRelations));

		// main test loop for 8 bits image
		for (loopCnt = 0; loopCnt < (MAX_TEST_LOOP); loopCnt++)
		{
			memset(pTrivialHist8Buffer, 0, histogram8BufSize);
			memset(pOptHist8Buffer, 0, histogram8OptBufSize);
			memset(pOptFinalHist8Buffer, 0, histogram8BufSize);

			statVal[0][loopCnt].cnt = loopCnt;
			QueryPerformanceCounter(&ts1);
			createTrivialHistogram8(pImageBuf8, DATA_SIZE_X, DATA_SIZE_Y, DATA_LINE_PITCH, pTrivialHist8Buffer);
			QueryPerformanceCounter(&ts2);
			statVal[0][loopCnt].iterClck.QuadPart = ts2.QuadPart - ts1.QuadPart;
		}

		// main test loop for 16 bits image
		for (loopCnt = 0; loopCnt < (MAX_TEST_LOOP); loopCnt++)
		{
			memset(pTrivialHist16Buffer, 0, histogram16BufSize);
			memset(pOptHist16Buffer, 0, histogram16OptBufSize);
			memset(pOptFinalHist16Buffer, 0, histogram16BufSize);

			statVal[1][loopCnt].cnt = loopCnt;
			QueryPerformanceCounter(&ts1);
			createTrivialHistogram16(pImageBuf16, DATA_SIZE_X, DATA_SIZE_Y, DATA_LINE_PITCH, pTrivialHist16Buffer);
			QueryPerformanceCounter(&ts2);
			statVal[1][loopCnt].iterClck.QuadPart = ts2.QuadPart - ts1.QuadPart;
		}

	} // if (nullptr != pImageBuf16 && nullptr != pImageBuf8 && nullptr != pTrivialHist16Buffer && nullptr != pTrivialHist8Buffer ...


	// free memory resources
	if (nullptr != pImageBuf16)
	{
		_aligned_free(pImageBuf16);
		pImageBuf16 = nullptr;
	}
	if (nullptr != pImageBuf8)
	{
		_aligned_free(pImageBuf8);
		pImageBuf8 = nullptr;
	}
	if (nullptr != pTrivialHist16Buffer)
	{
		_aligned_free(pTrivialHist16Buffer);
		pTrivialHist16Buffer = nullptr;
	}
	if (nullptr != pTrivialHist8Buffer)
	{
		_aligned_free(pTrivialHist8Buffer);
		pTrivialHist8Buffer = nullptr;
	}
	if (nullptr != pOptHist16Buffer)
	{
		_aligned_free(pOptHist16Buffer);
		pOptHist16Buffer = nullptr;
	}
	if (nullptr != pOptHist8Buffer)
	{
		_aligned_free(pOptHist8Buffer);
		pOptHist8Buffer = nullptr;
	}
	if (nullptr != pOptFinalHist16Buffer)
	{
		_aligned_free(pOptFinalHist16Buffer);
		pOptFinalHist16Buffer = nullptr;
	}
	if (nullptr != pOptFinalHist8Buffer)
	{
		_aligned_free(pOptFinalHist8Buffer);
		pOptFinalHist8Buffer = nullptr;
	}

	printf("Complete. Press <ENTER> for exit...\n");
	(void)getchar();

	return EXIT_SUCCESS;
}