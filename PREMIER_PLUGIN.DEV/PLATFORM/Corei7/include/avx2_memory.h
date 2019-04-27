#include <immintrin.h>
#include <Windows.h>

#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) avx2_mem_clean(void* __restrict pMem, const size_t memSize);
	void __declspec(dllexport) avx2_mem256_block_clean(void* __restrict pMem);
	void __declspec(dllexport) avx2_mem1024_block_clean(void* __restrict pMem);

#ifdef __cplusplus
}
#endif
