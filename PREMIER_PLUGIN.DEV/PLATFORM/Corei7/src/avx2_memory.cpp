#include "avx2_memory.h"

void __declspec(dllexport) avx2_mem_clean(void* __restrict pMem, const size_t memSize)
{

}


void __declspec(dllexport) avx2_mem256_block_clean(void* __restrict pMem)
{
	const __m256i zeroVector = _mm256_setzero_si256();
	__m256i* p = reinterpret_cast<__m256i* __restrict>(pMem);

	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector), p++;
	_mm256_store_si256(p, zeroVector);
}


void __declspec(dllexport) avx2_mem1024_block_clean(void* __restrict pMem)
{
	avx2_mem256_block_clean(pMem);
	avx2_mem256_block_clean(reinterpret_cast<void* __restrict>((reinterpret_cast<__m256i* __restrict>(pMem)) + 8));
	avx2_mem256_block_clean(reinterpret_cast<void* __restrict>((reinterpret_cast<__m256i* __restrict>(pMem)) + 16));
	avx2_mem256_block_clean(reinterpret_cast<void* __restrict>((reinterpret_cast<__m256i* __restrict>(pMem)) + 24));
}
