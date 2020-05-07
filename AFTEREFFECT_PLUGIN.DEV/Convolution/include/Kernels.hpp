#pragma once

#include <stdint.h>

void InitKernelsFactory(void);

template <typename T>
class IAbsrtactKernel
{
public:
	virtual ~IAbsrtactKernel() = default;
	virtual const bool LoadKernel(void) = 0;
	virtual const T* GetArray(void) = 0;
	virtual const uint32_t GetSize(void) = 0;
	virtual const T   GetFactor(void) = 0;
};


template <typename T>
class Sharp3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1
	};

	const uint32_t size = 3u;
	const T factor{ 1 };

public:
	Sharp3x3() = default;
	virtual ~Sharp3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Sharp5x5 : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		-1, -1, -1, -1, -1,
	    -1,  2,  2,  2, -1, 
	    -1,  2,  8,  2,  1,
		-1,  2,  2,  2, -1,
		-1, -1, -1, -1, -1
	};

	const uint32_t size = 5u;
	const float factor{ 1.f / 8.f };

public:
	Sharp5x5() = default;
	virtual ~Sharp5x5() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Blur3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		0, 1, 0,
		1, 1, 1,
		0, 1, 1
	};

	const uint32_t size = 3u;
	const float factor{ 1.f / 6.f };

public:
	Blur3x3() = default;
	virtual ~Blur3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Blur5x5 : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
		0, 0, 1, 0, 0
	};

	const uint32_t size = 5u;
	const float factor{ 1.f / 13.f };

public:
	Blur5x5() = default;
	virtual ~Blur5x5() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Sharpen3x3Factor : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		0, -2,  0,
       -2, 11, -2,
	    0, -2,  0
	};

	const uint32_t size = 3u;
	const float factor{ 1.f / 3.f };

public:
	Sharpen3x3Factor() = default;
	virtual ~Sharpen3x3Factor() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class IntenceSharpen3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		1,  1, 1,
	    1, -7, 1,
		1,  1, 1
	};

	const uint32_t size = 3u;
	const T factor{ 1 };

public:
	IntenceSharpen3x3() = default;
	virtual ~IntenceSharpen3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class EdgeDetection : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		-1, -1, -1,
		-1,  8, -1,
		-1, -1, -1
	};

	const uint32_t size = 3u;
	const T factor{ 1 };

public:
	EdgeDetection() = default;
	virtual ~EdgeDetection() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Edge45Degrees : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		-1,  0,  0,  0,  0,
		 0, -2,  0,  0,  0,
		 0,  0,  6,  0,  0,
		 0,  0,  0, -2,  0,
		 0,  0,  0,  0, -1
	};

	const uint32_t size = 5u;
	const T factor{ 1 };

public:
	Edge45Degrees() = default;
	virtual ~Edge45Degrees() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class EdgeHorizontal : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		0,  0,  0,  0,  0,
	    0,  0,  0,  0,  0,
	   -1, -1,  4, -1, -1,
	    0,  0,  0,  0,  0,
	    0,  0,  0,  0,  0
	};

	const uint32_t size = 5u;
	const T factor{ 1 };

public:
	EdgeHorizontal() = default;
	virtual ~EdgeHorizontal() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class EdgeVertical : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		0,  0, -1,  0,  0,
		0,  0, -1,  0,  0,
		0,  0,  4,  0,  0,
		0,  0, -1,  0,  0,
		0,  0, -1,  0,  0
	};

	const uint32_t size = 5u;
	const T factor{ 1 };

public:
	EdgeVertical() = default;
	virtual ~EdgeVertical() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Emboss : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		2,  0,  0,
	    0, -1,  0,
	    0,  0, -1
	};

	const uint32_t size = 3u;
	const T factor{ 1 };

public:
	Emboss() = default;
	virtual ~Emboss() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class IntenseEmboss : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		-1, -1, -1, -1,  0,
		-1, -1, -1,  0,  1,
		-1, -1,  0,  1,  1,
		-1,  0,  1,  1,  1,
		 0,  1,  1,  1,  1
	};

	const uint32_t size = 5u;
	const T factor{ 1 };

public:
	IntenseEmboss() = default;
	virtual ~IntenseEmboss() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Soften3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1
	};

	const uint32_t size = 3u;
	const float factor{ 1.f / 8.f };

public:
	Soften3x3() = default;
	virtual ~Soften3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Soften5x5 : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1
	};

	const uint32_t size = 5u;
	const float factor{ 1.f / 24.f };

public:
	Soften5x5() = default;
	virtual ~Soften5x5() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Gaussian3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		1, 2, 1,
		2, 4, 4,
		1, 2, 1
	};

	const uint32_t size = 3u;
	const float factor{ 1.f / 16.f };

public:
	Gaussian3x3() = default;
	virtual ~Gaussian3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Gaussian5x5 : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		2,  4,  5,  4, 2,
		4,  9, 12,  9, 4,
	    5, 12, 15, 12, 5,
		4,  9, 12,  9, 4,
		2,  4,  5,  4, 2
	};

	const uint32_t size = 5u;
	const float factor{ 1.f / 159.f };

public:
	Gaussian5x5() = default;
	virtual ~Gaussian5x5() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Laplacian3x3 : public IAbsrtactKernel<T>
{
private:
	const T kernel[9] =
	{
		0, -1,  0,
	   -1,  4, -1,
	    0, -1,  0
	};

	const uint32_t size = 3u;
	const T factor{ 1 };

public:
	Laplacian3x3() = default;
	virtual ~Laplacian3x3() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class Laplacian5x5 : public IAbsrtactKernel<T>
{
private:
	const T kernel[25] =
	{
		0,  0, -1,  0,  0,
		0, -1, -2, -1,  0,
	   -1, -2, 16, -2, -1,
	    0, -1, -2, -1,  0,
	    0,  0, -1,  0,  0,
	};

	const uint32_t size = 5u;
	const T factor{ 1 };

public:
	Laplacian5x5() = default;
	virtual ~Laplacian5x5() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class MotionBlur9x9 : public IAbsrtactKernel<T>
{
private:
	const T kernel[81] =
	{
		1, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 1, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 1, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 1
	};

	const uint32_t size = 9u;
	const float factor{ 1.0/18.f };

public:
	MotionBlur9x9() = default;
	virtual ~MotionBlur9x9() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class MotionBlurL2R9x9 : public IAbsrtactKernel<T>
{
private:
	const T kernel[81] =
	{
		1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1
	};

	const uint32_t size = 9u;
	const float factor{ 1.0 / 9.f };

public:
	MotionBlurL2R9x9() = default;
	virtual ~MotionBlurL2R9x9() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class MotionBlurR2L9x9 : public IAbsrtactKernel<T>
{
private:
	const T kernel[81] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0
	};

	const uint32_t size = 9u;
	const float factor{ 1.0 / 9.f };

public:
	MotionBlurR2L9x9() = default;
	virtual ~MotionBlurR2L9x9() = default;
	const bool     LoadKernel(void) { return true; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};


template <typename T>
class CustomKernel : public IAbsrtactKernel<T>
{
private:
	T* kernel;
	uint32_t size = 0u;
	float factor{ 1.0 / 9.f };

public:
	CustomKernel() = default;
	virtual ~CustomKernel() = default;
	const bool     LoadKernel(void) { return false; }
	const T*       GetArray(void)   { return kernel; }
	const uint32_t GetSize(void)    { return size; }
	const T        GetFactor(void)  { return factor; }
};