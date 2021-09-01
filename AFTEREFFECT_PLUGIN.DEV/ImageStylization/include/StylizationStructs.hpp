#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

constexpr int circular_size = 3;
constexpr int hist_size_H = 360;
constexpr int hist_size_S = 256;
constexpr int hist_size_I = 256;
constexpr unsigned int max_buf_idx = 4u;


typedef struct ImageStyleTmpStorage {
	std::mutex guard_buffer;
	std::atomic<size_t> bufMemSize;
	float* __restrict pStorage1;

	ImageStyleTmpStorage::ImageStyleTmpStorage(void)
	{
		pStorage1 = nullptr;
		bufMemSize = 0ul;
	}
}ImageStyleTmpStorage;


constexpr unsigned int CartoonEffectBuf_size = static_cast<unsigned int>(sizeof(ImageStyleTmpStorage));

ImageStyleTmpStorage* alloc_temporary_buffers(const size_t& mem_size) noexcept;
void free_temporary_buffers(ImageStyleTmpStorage* pStr) noexcept;
bool test_temporary_buffers(ImageStyleTmpStorage* pStr, const size_t& mem_size) noexcept;


typedef float tPointilismBrush[11][11];

const tPointilismBrush& getPointilismBrush1(void);
const tPointilismBrush& getPointilismBrush2(void);
const tPointilismBrush& getPointilismBrush3(void);
const tPointilismBrush& getPointilismBrush4(void);