#include "StylizationStructs.hpp"

std::atomic<int> gCartoonBuffers{};
std::mutex globalProtect;
constexpr int maxCartoonBuffers = 16;


CartoonEffectBuf* alloc_cartoon_effect_buffers(int width, int height) noexcept
{
	{
		std::lock_guard<std::mutex> global_lock(globalProtect);
		gCartoonBuffers++;
		if (gCartoonBuffers > maxCartoonBuffers)
		{
			gCartoonBuffers--;
			return nullptr;
		}
	}

	/* allocate structure itself */
	CartoonEffectBuf* pBuf = new CartoonEffectBuf;

	const size_t memSize = width * height * sizeof(float);
	constexpr size_t alignment = CACHE_LINE;
	const size_t bufSize = CreateAlignment(memSize, alignment);

	if (nullptr != pBuf)
	{
		float* __restrict pH = nullptr;
		float* __restrict pS = nullptr;
		float* __restrict pI = nullptr;

		/* allocate buffers */
		pH = reinterpret_cast<float* __restrict>(_aligned_malloc(bufSize, alignment));
		pS = reinterpret_cast<float* __restrict>(_aligned_malloc(bufSize, alignment));
		pI = reinterpret_cast<float* __restrict>(_aligned_malloc(bufSize, alignment));

		if ((nullptr != pH) && (nullptr != pS) && (nullptr != pI))
		{
#ifdef _DEBUG
			memset(pH, 0, bufSize);
			memset(pS, 0, bufSize);
			memset(pI, 0, bufSize);

			memset(&pBuf->histBuf, 0, sizeof(pBuf->histBuf));
#endif /* _DEBUG */
			pBuf->bufH = pH;
			pBuf->bufS = pS;
			pBuf->bufI = pI;
			pBuf->bufMemSize = bufSize;
		}
		else
		{
#ifdef _DEBUG
			pBuf->bufH = pBuf->bufS = pBuf->bufI = nullptr;
			pBuf->bufMemSize = 0ul;
#endif
			if (nullptr != pH)
			{
				_aligned_free(pH);
				pH = nullptr;
			}
			if (nullptr != pS)
			{
				_aligned_free(pS);
				pS = nullptr;
			}
			if (nullptr != pI)
			{
				_aligned_free(pI);
				pI = nullptr;
			}

			delete pBuf;
			pBuf = nullptr;
		}
	}

	return pBuf;
}


void free_cartoon_effect_buffers(CartoonEffectBuf* pStr) noexcept
{
	if (nullptr != pStr)
	{
		gCartoonBuffers--;

		if (nullptr != pStr->bufH)
		{
			_aligned_free(pStr->bufH);
			pStr->bufH = nullptr;
		}
		if (nullptr != pStr->bufS)
		{
			_aligned_free(pStr->bufS);
			pStr->bufS = nullptr;
		}
		if (nullptr != pStr->bufI)
		{
			_aligned_free(pStr->bufI);
			pStr->bufI = nullptr;
		}
#ifdef _DEBUG
		memset(&pStr->histBuf, 0, sizeof(pStr->histBuf));
#endif
		delete pStr;
		pStr = nullptr;
	}
	return;
}


bool test_cartoon_effect_buffers(CartoonEffectBuf* pStr, int width, int height) noexcept
{
	bool bTestResult = false;

	if (nullptr != pStr)
	{
		const size_t actualSize = width * height * sizeof(float);
		if (pStr->bufMemSize < actualSize)
		{
			constexpr size_t alignment = CACHE_LINE;
			const size_t bufSize = CreateAlignment(actualSize, alignment);

			std::lock_guard<std::mutex> lock(pStr->guard_buffer);

			/* make realloc */
			float* __restrict pH = reinterpret_cast<float* __restrict>(_aligned_realloc(reinterpret_cast<void*>(pStr->bufH), bufSize, alignment));
			float* __restrict pS = reinterpret_cast<float* __restrict>(_aligned_realloc(reinterpret_cast<void*>(pStr->bufS), bufSize, alignment));
			float* __restrict pI = reinterpret_cast<float* __restrict>(_aligned_realloc(reinterpret_cast<void*>(pStr->bufI), bufSize, alignment));

			if ((nullptr != pH) && (nullptr != pS) && (nullptr != pI))
			{
				pStr->bufH = pH;
				pStr->bufS = pS;
				pStr->bufI = pI;
				pStr->bufMemSize = bufSize;

				bTestResult = true;
			}
		}
	}

	return bTestResult;
}