#pragma once

#include <iostream>
#include <immintrin.h>

#ifdef _DEBUG
#include <mutex>

std::mutex coutProtect;
#endif

namespace AVX2
{
	namespace Debug
	{
		template<class T>
		static inline void Avx2RegLog(const __m256i& value)
		{
			constexpr size_t n = sizeof(__m256i) / sizeof(T);
			T buffer[n];
			_mm256_storeu_si256((__m256i*)buffer, value);
#ifdef _DEBUG
			std::lock_guard<std::mutex> global_lock(coutProtect);
#endif
			if (1ull == sizeof(T))
			{
				for (int i = 0; i < n; i++)
					std::cout << static_cast<int>(buffer[i]) << " ";
			}
			else
			{
				for (int i = 0; i < n; i++)
					std::cout << buffer[i] << " ";
			}
			std::cout << std::endl;
		}

		template<class T>
		static const T* Avx2RegGet(const __m256i& value)
		{
			constexpr size_t n = sizeof(__m256i) / sizeof(T);
			T buffer[n];
			_mm256_storeu_si256((__m256i*)buffer, value);
			return buffer;
		}

	} /* namespace Debug */
}/* AVX2 */