#include "ImageLabAnisotropicDiffusion.h"


void algMemStorageFree (AlgMemStorage& algMemStorage)
{
	algMemStorage.memBytesSize = 0;
	if (nullptr != algMemStorage.pTmp1)
	{
		_aligned_free(algMemStorage.pTmp1);
		algMemStorage.pTmp1 = nullptr;
	}

	if (nullptr != algMemStorage.pTmp2)
	{
		_aligned_free(algMemStorage.pTmp2);
		algMemStorage.pTmp2 = nullptr;
	}
	return;
}


bool algMemStorageRealloc (const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage& algMemStorage)
{
	void* p1 = nullptr;
	void* p2 = nullptr;
	bool bResult = false;

	algMemStorageFree(algMemStorage);
	constexpr csSDK_int32 pixelSize = sizeof(float);
	constexpr size_t memBufferAlignment = static_cast<size_t>(CPU_PAGE_SIZE);
	const size_t memBytesSize = pixelSize * width * height;
	const size_t alignedBytesSize = CreateAlignment (memBytesSize, memBufferAlignment);

	p1 = _aligned_malloc (alignedBytesSize, memBufferAlignment);
	p2 = _aligned_malloc (alignedBytesSize, memBufferAlignment);

	if (nullptr != p1 && nullptr != p2)
	{
#ifdef _DEBUG
		/* cleanup allocated memory for DBG purpose only */
		__VECTOR_ALIGNED__
		memset(p1, 0, alignedBytesSize);
		__VECTOR_ALIGNED__
		memset(p2, 0, alignedBytesSize);
#endif
		algMemStorage.memBytesSize = alignedBytesSize;
		algMemStorage.pTmp1 = p1;
		algMemStorage.pTmp2 = p2;
		bResult = true;
	}
	else
	{
		algMemStorageFree(algMemStorage);
		bResult = false;
	}

	return bResult;
}
