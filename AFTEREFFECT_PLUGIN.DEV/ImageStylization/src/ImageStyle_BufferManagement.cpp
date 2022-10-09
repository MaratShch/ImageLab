#include "StylizationStructs.hpp"

std::atomic<int> gBuffers{};
std::mutex globalProtect;
constexpr int maxBuffers = 16;
constexpr size_t BufAlignment = CPU_PAGE_SIZE;


ImageStyleTmpStorage* alloc_temporary_buffers (const size_t& mem_size) noexcept
{
	std::lock_guard<std::mutex> global_lock(globalProtect);
	gBuffers++;
	if (gBuffers > maxBuffers)
	{
		gBuffers--;
		return nullptr;
	}

	/* allocate structure itself */
	ImageStyleTmpStorage* pBuf = new ImageStyleTmpStorage;

	const size_t& memSize = mem_size;
	const size_t  bufSize = CreateAlignment(memSize, BufAlignment);

	if (nullptr != pBuf)
	{
		float* __restrict pStorage1 = nullptr;

		/* allocate buffers */
		pStorage1 = reinterpret_cast<float* __restrict>(_aligned_malloc(bufSize, BufAlignment));

		if (nullptr != pStorage1)
		{
#ifdef _DEBUG
			memset(pStorage1, 0, bufSize);
#endif /* _DEBUG */
			pBuf->pStorage1 = pStorage1;
			pBuf->bufMemSize = bufSize;
		}
		else
		{
#ifdef _DEBUG
			pBuf->pStorage1 = nullptr;
			pBuf->bufMemSize = 0ul;
#endif
			if (nullptr != pStorage1)
			{
				_aligned_free(pStorage1);
				pStorage1 = nullptr;
			}
			delete pBuf;
			pBuf = nullptr;
		}
	}

	return pBuf;
}


void free_temporary_buffers (ImageStyleTmpStorage* pStr) noexcept
{
	if (nullptr != pStr)
	{
		gBuffers--;

		if (nullptr != pStr->pStorage1)
		{
			std::lock_guard<std::mutex> lock(pStr->guard_buffer);
#ifdef _DEBUG
			memset(pStr->pStorage1, 0, pStr->bufMemSize);
#endif
			_aligned_free(pStr->pStorage1);
			pStr->pStorage1 = nullptr;
			pStr->bufMemSize = 0ul;
		}
		delete pStr;
		pStr = nullptr;
	}
	return;
}


bool test_temporary_buffers(ImageStyleTmpStorage* pStr, const size_t& mem_size) noexcept
{
	bool bTestResult = false;

	if (nullptr != pStr)
	{
		const size_t actualSize = mem_size;
		if (pStr->bufMemSize < actualSize)
		{
			std::lock_guard<std::mutex> lock(pStr->guard_buffer);

			const size_t bufSize = CreateAlignment(actualSize, BufAlignment);

			/* make realloc */
			float* __restrict pStorage1 = reinterpret_cast<float* __restrict>(_aligned_realloc(reinterpret_cast<void*>(pStr->pStorage1), bufSize, BufAlignment));

			if (nullptr != pStorage1)
			{
				pStr->pStorage1 = pStorage1;
				pStr->bufMemSize = bufSize;
				bTestResult = true;
			}
		}
		else
		{
			bTestResult = true;
		}
	}

	return bTestResult;
}