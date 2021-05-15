#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

constexpr unsigned int hist_size_H = 360u;
constexpr unsigned int hist_size_S = 256u;
constexpr unsigned int hist_size_I = 256u;
constexpr unsigned int max_buf_idx = 4u;

typedef struct CartoonEffectParamStr {

	unsigned int histH[hist_size_H];
	unsigned int histS[hist_size_S];
	unsigned int histI[hist_size_I];
}CartoonEffectParamStr;

typedef struct CartoonEffectBuf {
	std::mutex guard_buffer;
	float* __restrict bufH;
	float* __restrict bufS;
	float* __restrict bufI;
	size_t bufMemSize;
	CartoonEffectParamStr histBuf;

	CartoonEffectBuf::CartoonEffectBuf(void)
	{
		bufH = bufS = bufI = nullptr;
		bufMemSize = 0ul;
		memset(&histBuf, 0, sizeof(histBuf));
	}
}CartoonEffectBuf;

constexpr unsigned int CartoonEffectParamStr_size = static_cast<unsigned int>(sizeof(CartoonEffectParamStr));
constexpr unsigned int CartoonEffectBuf_size = static_cast<unsigned int>(sizeof(CartoonEffectBuf));

CartoonEffectBuf* alloc_cartoon_effect_buffers(int width, int height) noexcept;
void free_cartoon_effect_buffers(CartoonEffectBuf* pStr) noexcept;
bool test_cartoon_effect_buffers(CartoonEffectBuf* pStr, int width, int height) noexcept;
