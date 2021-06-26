#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

constexpr int circular_size = 3;
constexpr int hist_size_H = 360;
constexpr int hist_size_S = 256;
constexpr int hist_size_I = 256;
constexpr unsigned int max_buf_idx = 4u;


typedef struct ImageStyleTmpStorage {
	std::mutex guard_buffer;
	float* __restrict pStorage1;
	size_t bufMemSize;

	ImageStyleTmpStorage::ImageStyleTmpStorage(void)
	{
		pStorage1 = nullptr;
		bufMemSize = 0ul;
	}
}ImageStyleTmpStorage;

typedef struct CostData
{
	float cost;
	int32_t imin1;
	int32_t imin2;
	int32_t typemerging;

	CostData::CostData(void)
	{
		cost = 0.f;
		imin1 = imin2 = typemerging = 0;
	}
} CostData;

constexpr unsigned int CartoonEffectBuf_size = static_cast<unsigned int>(sizeof(ImageStyleTmpStorage));

ImageStyleTmpStorage* alloc_temporary_buffers(const size_t& mem_size) noexcept;
void free_temporary_buffers(ImageStyleTmpStorage* pStr) noexcept;
bool test_temporary_buffers(ImageStyleTmpStorage* pStr, const size_t& mem_size) noexcept;
