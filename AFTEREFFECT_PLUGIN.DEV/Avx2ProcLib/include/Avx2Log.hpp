#pragma once

#include <iostream>
#include <immintrin.h>


namespace AVX2
{
	namespace Debug
	{
		template<class T>
		typename std::enable_if<std::is_integral<T>::value>::type Avx2RegLog(const __m256i& value) noexcept
		{
			constexpr size_t n = sizeof(__m256i) / sizeof(T);
			__declspec(align(32)) T buffer[n];
			_mm256_storeu_si256((__m256i*)buffer, value);
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
		typename std::enable_if<std::is_floating_point<T>::value>::type Avx2RegLog(const __m256& value) noexcept
		{
			constexpr size_t n = sizeof(__m256) / sizeof(T);
			__declspec(align(32)) T buffer[n];
			_mm256_storeu_ps(buffer, value);
			for (int i = 0; i < n; i++)
				std::cout << buffer[i] << " ";
			std::cout << std::endl;
		}

	} /* namespace Debug */

}/* AVX2 */