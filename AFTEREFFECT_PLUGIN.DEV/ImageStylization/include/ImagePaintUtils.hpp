#pragma once

#include "CommonPixFormat.hpp"
#include <malloc.h>
#include <stdio.h>
#include <vector>
#include <map>


template <typename T>
class SparseMatrix
{
public:
	SparseMatrix ()
	{
		n_columns = 0;
		columns.clear();
		columns.resize(n_columns);
	}
	explicit SparseMatrix (A_long columns)
	{
		n_columns = columns;
		columns.clear();
		columns.resize(n_columns);
	}
	
	A_long nColumns() const noexcept { return n_columns; }
	
	const std::map<A_long, T> & get_column (const A_long& col) const noexcept
	{
		return columns[col];
	}

	T& operator() (const A_long& row, const A_long& col) noexcept
	{ 
		return columns[col][row];
	}

	T operator() (const A_long& row, const A_long& col) const noexcept
	{
		auto it = columns[col].find(row);
		return ((it == columns[col].end()) ? 0 : it->second);
	}

private:
	A_long n_columns;
	std::vector<std::map<A_long, T>> columns;
};


bool bw_image2cocircularity_graph
(
	const float* __restrict im,
	SparseMatrix<float>& S,
	      float* __restrict im_anisotropy,
	const A_long& width,
	const A_long& height,
	const A_long& pitch,
	const float&  sigma,
	const float&  coCirc,
	const float&  coCone,
	const A_long& p
) noexcept;

bool bw_image2cocircularity_graph
(
	const float* __restrict im,
	std::unique_ptr<SparseMatrix<float>>& S,
	      float* __restrict im_anisotropy,
	A_long width,
	A_long height,
	A_long pitch,
	float sigma,
	float coCirc,
	float coCone,
	A_long p
) noexcept;

void linear_gradient_gray
(
	const float* __restrict im,
	std::unique_ptr<float[]>& gX,
	std::unique_ptr<float[]>& gY,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& pitch
) noexcept;

void structure_tensors0
(
	std::unique_ptr<float[]>& gX,
	std::unique_ptr<float[]>& gY,
	const A_long& sizeX,
	const A_long& sizeY,
	std::unique_ptr<float[]>& A,
	std::unique_ptr<float[]>& B,
	std::unique_ptr<float[]>& C
) noexcept;

void smooth_structure_tensors
(
	std::unique_ptr<float[]>& A,
	std::unique_ptr<float[]>& B,
	std::unique_ptr<float[]>& C,
	const float& sigma,
	const A_long& sizeX,
	const A_long& sizeY,
	std::unique_ptr<float[]>& A_reg,
	std::unique_ptr<float[]>& B_reg,
	std::unique_ptr<float[]>& C_reg
) noexcept;

void convolution
(
	std::unique_ptr<float[]>& imIn,
	std::unique_ptr<float[]>& gKernel,
	const float& sigma,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& radius,
	std::unique_ptr<float[]>& imOut
) noexcept;

void diagonalize_structure_tensors
(
	const std::unique_ptr<float[]>& a,
	const std::unique_ptr<float[]>& b,
	const std::unique_ptr<float[]>& c,
	A_long sizeX,
	A_long sizeY,
	std::unique_ptr<float[]>& lambda1,
	std::unique_ptr<float[]>& lambda2,
	std::unique_ptr<float[]>& eigvect2_x,
	std::unique_ptr<float[]>& eigvect2_y,
	std::unique_ptr<float[]>& anisotropy
) noexcept;


template <typename T>
inline T* allocTmpBuffer (const A_long& height, const A_long& pitch, T** procBuf) noexcept
{
	const A_long line_pitch = (pitch < 0 ? -pitch : pitch);
	const size_t elemNumber = height * line_pitch;
	T* rawPtr = new T[elemNumber];
#ifdef _DEBUG
	memset(rawPtr, 0, elemNumber * sizeof(T));
#endif
	if (nullptr != procBuf)
		*procBuf = rawPtr + ((pitch < 0) ? elemNumber - line_pitch : 0);

	return rawPtr;
}

template <typename T>
inline void freeTmpBuffer (T* ptr) noexcept
{
	if (nullptr != ptr)
	{
		delete[] ptr;
		ptr = nullptr;
	}
	return;
}


/* =================================================================== */
/* simple and rapid (non precise) convert RGB to Black-and-White image */
/* =================================================================== */
template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline void Color2Bw
(
	const T* __restrict pSrc,
	float*   __restrict pDst,
	const A_long&       width,
	const A_long&       height,
	const A_long&       pitch
) noexcept
{
	constexpr float reciproc3{ 1.f / 3.f };
	A_long j, i;
	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const A_long idx{ j * pitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = (static_cast<float>(pSrc[idx + i].R) + static_cast<float>(pSrc[idx + i].G) + static_cast<float>(pSrc[idx + i].B)) * reciproc3;
	}
	return;
}


template <class T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void Color2Bw
(
	const T* __restrict pSrc,
	float*   __restrict pDst,
	const A_long&       width,
	const A_long&       height,
	const A_long&       pitch
) noexcept
{
	__VECTOR_ALIGNED__
	for (A_long j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * pitch;
		float*   __restrict pDstLine = pDst + j * pitch;
		for (A_long i = 0; i < width; i++)
			pSrcLine[i] = static_cast<float>(pSrc[i].Y);
	}
	return;
}
