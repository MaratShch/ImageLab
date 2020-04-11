#pragma once

#include <utility>

template<typename T>
constexpr T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
constexpr T MAX(T a, T b) { return ((a > b) ? a : b); }

template <typename T>
static inline void swapEx(T& a, T& b) // a != b && a , b = integral types
{
	a = a ^ b;
	b = a ^ b;
	a = a ^ b;
}

template <typename T>
static inline void gnomesort(T* l, T* r)
{
	T* i = l;
	while (i < r)
	{
		if (i == l || *(i - 1) <= *i)i++;
		else std::swap (*(i - 1), *i), i--;
	}
}


template <typename T>
static inline void selectionsort(T* l, T* r)
{
	for (T* i = l; i < r; i++)
	{
		T minz = *i, *ind = i;
		for (T* j = i + 1; j < r; j++)
		{
			if (*j < minz) minz = *j, ind = j;
		}
		std::swap (*i, *ind);
	}
}

template <typename T>
static inline void insertionsort(T* l, T* r)
{
	for (T* i = l + 1; i < r; i++)
	{
		T* j = i;
		while (j > l && *(j - 1) > *j)
		{
			std::swap (*(j - 1), *j);
			j--;
		}
	}
}

