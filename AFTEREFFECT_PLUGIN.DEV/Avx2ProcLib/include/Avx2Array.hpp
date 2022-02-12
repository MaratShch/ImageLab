#pragma once

#include <cstdint>
#include <immintrin.h>

namespace AVX2
{

	template <typename T> struct Array
	{
		T* data;
		size_t size;
	};

	template <typename T, size_t s> struct FixArray
	{
		T data[s];
	};

	typedef Array<uint8_t> Array8u;

};