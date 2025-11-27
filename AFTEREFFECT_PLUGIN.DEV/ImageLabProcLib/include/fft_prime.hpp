#pragma once 

#include <cmath>
#include <vector>

namespace FourierTransform
{
	
inline std::vector<int32_t> prime (const int32_t N)
{
	std::vector<int32_t> factors{};
	int32_t tmp_N = N;

// 1. Prioritize largest powers of 2 for which you have butterflies
	while (0 == tmp_N % 16)
	{
		factors.push_back(16);
		tmp_N /= 16; 
	}		

	while (0 == tmp_N % 8)
	{
		factors.push_back(8);
		tmp_N /= 8; 
	}		

	while (0 == tmp_N % 4)
	{
		factors.push_back(4);
		tmp_N /= 4; 
	}		

	while (0 == tmp_N % 2)
	{
		factors.push_back(2);
		tmp_N /= 2; 
	}		

// 2. Small odd primes
	while (0 == tmp_N % 7)
	{ 
		factors.push_back(7);
		tmp_N /= 7;
	}
	
	while (0 == tmp_N % 5)
	{
		factors.push_back(5);
		tmp_N /= 5;
	}
	
	while (0 == tmp_N % 3)
	{
		factors.push_back(3);
		tmp_N /= 3;
	}
	
	// 3. Remainder (handled by CZT)
	if (tmp_N > 1)
	{
		factors.push_back(tmp_N);
	}
	
	return factors;
}	

}