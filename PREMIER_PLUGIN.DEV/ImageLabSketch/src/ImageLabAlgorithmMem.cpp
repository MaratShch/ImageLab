#include "ImageLabSketch.h"


void algMemStorageFree (AlgMemStorage* pAlgMemStorage)
{
	if (nullptr != pAlgMemStorage)
	{
		pAlgMemStorage->bytesSize = 0;
		if (nullptr != pAlgMemStorage->pBuf1)
		{
			_aligned_free(pAlgMemStorage->pBuf1);
			pAlgMemStorage->pBuf1 = nullptr;
		}

		if (nullptr != pAlgMemStorage->pBuf2)
		{
			_aligned_free(pAlgMemStorage->pBuf2);
			pAlgMemStorage->pBuf2 = nullptr;
		}
	}
	return;
}


bool algMemStorageRealloc (const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage* pAlgMemStorage)
{
	bool bResult = false;

	if (nullptr != pAlgMemStorage)
	{
		void* __restrict p1 = nullptr;
		void* __restrict p2 = nullptr;

		algMemStorageFree(pAlgMemStorage);
		constexpr csSDK_int32 pixelSize = sizeof(float);
		constexpr size_t memBufferAlignment = static_cast<size_t>(CPU_PAGE_SIZE);
		const size_t memBytesSize = pixelSize * width * height;
		const size_t alignedBytesSize = CreateAlignment(memBytesSize, memBufferAlignment);

		p1 = _aligned_malloc(alignedBytesSize, memBufferAlignment);
		p2 = _aligned_malloc(alignedBytesSize, memBufferAlignment);

		if (nullptr != p1 && nullptr != p2)
		{
#ifdef _DEBUG
			/* cleanup allocated memory for DBG purpose only */
			__VECTOR_ALIGNED__
			memset(p1, 0, alignedBytesSize);
			__VECTOR_ALIGNED__
			memset(p2, 0, alignedBytesSize);
#endif
			pAlgMemStorage->bytesSize = alignedBytesSize;
			pAlgMemStorage->pBuf1 = p1;
			pAlgMemStorage->pBuf2 = p2;
			bResult = true;
		}
		else
		{
			algMemStorageFree(pAlgMemStorage);
			bResult = false;
		}
	}
	return bResult;
}
