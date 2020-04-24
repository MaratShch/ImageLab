#pragma once

#include <type_traits>
#include <new>

template <typename T1, typename T2>
static inline const float g_function (const T1 x, const T2 k) /* T1, T2 - integral or floating point only types */
{
	const float div = static_cast<float>(x) / static_cast<float>(k);
	return (1.0f / (1.0f + div * div));
}


