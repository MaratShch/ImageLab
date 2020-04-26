#pragma once

#include "Common.hpp"

template <typename T>
class Matrix3x3
{
public:
	Matrix3x3(T arg, ...) {matrix = arg};
	virtual ~Matrix3x3() { ; }

private:
	T matrix[9];
};
