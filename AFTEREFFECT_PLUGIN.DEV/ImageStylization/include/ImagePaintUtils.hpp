#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"
#include "ColorTransformMatrix.hpp"
#include <malloc.h>
#include <stdio.h>
#include <vector>
#include <map>


template <typename T>
class SparseMatrix
{
public:
	CLASS_NON_COPYABLE(SparseMatrix);
	CLASS_NON_MOVABLE (SparseMatrix);

	SparseMatrix ()
	{
		n_columns = 0;
		m_columns.clear();
		m_columns.resize(n_columns);
		return;
	}

	explicit SparseMatrix (A_long columns)
	{
		n_columns = columns;
		m_columns.clear();
		m_columns.resize(n_columns);
		return;
	}

	A_long nColumns() const noexcept { return n_columns; }
	
	const std::map<A_long, T>& get_column (const A_long& col) const noexcept
	{
		return m_columns[col];
	}

	T& operator() (const A_long& row, const A_long& col) noexcept
	{ 
		return m_columns[col][row];
	}

	T operator() (const A_long& row, const A_long& col) const noexcept
	{
		auto it = m_columns[col].find(row);
		return ((it == m_columns[col].end()) ? 0 : it->second);
	}

private:
	A_long n_columns;
	std::vector<std::map<A_long, T>> m_columns;
};


bool bw_image2cocircularity_graph_impl
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

void compute_adjacency_matrix
(
	SparseMatrix<float>& S,
	std::unique_ptr<float[]>& eigvect2_x,
	std::unique_ptr<float[]>& eigvect2_y,
	A_long p,
	const A_long sizeX,
	const A_long sizeY,
	const float thresh_cocirc,
	const float thresh_cone
) noexcept;


void pixel_list
(
	A_long* __restrict row_list,
	A_long* __restrict col_list,
	const A_long& i_min1,
	const A_long& i_min2,
	const A_long& i_max1,
	const A_long& i_max2,
	const A_long& j_min1,
	const A_long& j_min2,
	const A_long& j_max1,
	const A_long& j_max2
) noexcept;


A_long count_sparse_matrix_non_zeros
(
	std::unique_ptr<SparseMatrix<float>>& S,
	const A_long& n_col
) noexcept;


void sparse_matrix_to_arrays
(
	std::unique_ptr<SparseMatrix<float>>& S,
	std::unique_ptr<A_long[]>& I,
	std::unique_ptr<A_long[]>& J,
	std::unique_ptr<float []>& W,
	A_long n_col
) noexcept;

A_long  morpho_open
(
	float*  __restrict imIn,
	float*  __restrict imOut,
	std::unique_ptr<float []>& w,
	std::unique_ptr<A_long[]>& i,
	std::unique_ptr<A_long[]>& j,
	A_long it,
	A_long nonZeros,
	A_long sizeX,
	A_long sizeY,
	float  normalizer = 255.0f
) noexcept;

int erode_max_plus_symmetric_iterated
(
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const float*  __restrict imIn,
	float*  __restrict imOut[],
	const A_long& k,
	const A_long& n_lines,
	float** pOut = nullptr,
	const float& normalizer = 255.0f,
	const A_long& frameSize = 0
) noexcept;

bool erode_max_plus_symmetric
(
	const float* __restrict imIn,
	float*  __restrict imOut,
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const A_long& n_lines,
	const float& normalizer = 255.0f,
	const A_long& frameSize = 0
) noexcept;

int dilate_max_plus_symmetric_iterated
(
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const float*  __restrict imIn,
	float*  __restrict imOut[],
	const A_long& k,
	const A_long& n_lines,
	float** pOut,
	const float& normalizer = 255.0f,
	const A_long& frameSize = 0
) noexcept;

bool dilate_max_plus_symmetric
(
	const float* __restrict imIn,
	float*  __restrict imOut,
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const A_long& n_lines,
	const float& normalizer = 255.0f,
	const A_long& frameSize = 0
) noexcept;




inline float* allocTmpBuffer (const A_long& height, const A_long& pitch, float** procBuf = nullptr) noexcept
{
	const A_long line_pitch = FastCompute::Abs(pitch);
	const size_t elemNumber = height * line_pitch;
	float* rawPtr = new float[elemNumber];
#ifdef _DEBUG
	memset(rawPtr, 0, elemNumber * sizeof(float));
#endif
	if (nullptr != procBuf)
		*procBuf = rawPtr + ((pitch < 0) ? elemNumber - line_pitch : 0);

	return rawPtr;
}

inline void freeTmpBuffer (float* ptr) noexcept
{
	if (nullptr != ptr)
	{
		delete[] ptr;
		ptr = nullptr;
	}
	return;
}


template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline void Color2YUV
(
	const T* __restrict pSrc,
	float*   __restrict pY,
	float*   __restrict pU,
	float*   __restrict pV,
	const A_long&       width,
	const A_long&       height,
	const A_long&       src_pitch,
	const A_long&       tmp_pitch,
	const A_long&       addendum = 0
) noexcept
{
	const float* pRgb2Yuv = RGB2YUV[BT709];

	for (A_long j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_pitch;
		float*   __restrict pYLine = pY + j * tmp_pitch;
		float*   __restrict pULine = pU + j * tmp_pitch;
		float*   __restrict pVLine = pV + j * tmp_pitch;

		for (A_long i = 0; i < width; i++)
		{
			pYLine[i] = static_cast<float>(pSrcLine[i].R) * pRgb2Yuv[0] + static_cast<float>(pSrcLine[i].G) * pRgb2Yuv[1] + static_cast<float>(pSrcLine[i].B) * pRgb2Yuv[2];
			pULine[i] = static_cast<float>(pSrcLine[i].R) * pRgb2Yuv[3] + static_cast<float>(pSrcLine[i].G) * pRgb2Yuv[4] + static_cast<float>(pSrcLine[i].B) * pRgb2Yuv[5] + addendum;
			pVLine[i] = static_cast<float>(pSrcLine[i].R) * pRgb2Yuv[6] + static_cast<float>(pSrcLine[i].G) * pRgb2Yuv[7] + static_cast<float>(pSrcLine[i].B) * pRgb2Yuv[8] + addendum;
		} /* for (A_long i = 0; i < width; i++) */
	} /* for (A_long j = 0; j < height; j++) */
	
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
	const A_long&       src_pitch,
	const A_long&       tmp_pitch 
) noexcept
{
	constexpr float reciproc3{ 1.f / 3.f };
	for (A_long j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_pitch;
		float*   __restrict pDstLine = pDst + j * tmp_pitch;
		for (A_long i = 0; i < width; i++)
		{
			pDstLine[i] = (static_cast<float>(pSrcLine[i].R) + static_cast<float>(pSrcLine[i].G) + static_cast<float>(pSrcLine[i].B)) * reciproc3;
		}
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
	const A_long&       src_pitch,
	const A_long&       tmp_pitch
) noexcept
{
	for (A_long j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * src_pitch;
		float*   __restrict pDstLine = pDst + j * tmp_pitch;
		for (A_long i = 0; i < width; i++)
			pSrcLine[i] = static_cast<float>(pSrc[i].Y);
	}
	return;
}
