#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImagePaintUtils.hpp"
#include <cmath>

void linear_gradient_gray
(
	const float* __restrict im,
	std::unique_ptr<float[]>& gX,
	std::unique_ptr<float[]>& gY,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& pitch
) noexcept
{
	A_long i, j;
	const A_long shortSizeX {sizeX - 1};
	const A_long shortSizeY {sizeY - 1};

	for (j = 0; j < sizeY; j++)
	{
		const float* __restrict pSrc = im + j * pitch;
		      float* __restrict gx = gX.get() + j * sizeX;
		/* first pixel - horizontal gradient */
		gx[0] = pSrc[1] - pSrc[0];

		/* horizontal gradient */
		for (i = 1; i < shortSizeX; i++)
			gx[i] = 0.5f * (pSrc[i + 1] - pSrc[i - 1]);

		/* last pixel - horizontal gradient */
		gx[i] = pSrc[i] - pSrc[i - 1];
	}

	{
		/* first line - vertical gradient */
		j = 0;
		const float* __restrict pSrcLine0 = im;
		const float* __restrict pSrcLine1 = im + pitch;
		float* __restrict gy = gY.get() + j * sizeX;

		for (i = 0; i < sizeX; i++)
			gy[i] = pSrcLine1[i] - pSrcLine0[i];
	}

	for (j = 1; j < shortSizeY; j++)
	{
		/* vertical gradient  */
		const float* __restrict pSrcLine0 = im + (j - 1) * pitch;
		const float* __restrict pSrcLine2 = im + (j + 1) * pitch;
		float* __restrict gy = gY.get() + j * sizeX;

		for (i = 0; i < sizeX; i++)
			gy[i] = 0.5f * (pSrcLine2[i] - pSrcLine0[i]);
	}

	{
		/* last line - vertical gradient */
		const float* __restrict pSrcLine0 = im + (j - 1) * pitch;
		const float* __restrict pSrcLine1 = im + j * pitch;
		float* __restrict gy = gY.get() + j * sizeX;

		for (i = 0; i < sizeX; i++)
			gy[i] = pSrcLine1[i] - pSrcLine0[i];
	}

	return;
}


void structure_tensors0
(
	std::unique_ptr<float[]>& gX,
	std::unique_ptr<float[]>& gY,
	const A_long& sizeX, 
	const A_long& sizeY,
	std::unique_ptr<float[]>& A,
	std::unique_ptr<float[]>& B,
	std::unique_ptr<float[]>& C
) noexcept
{
	A_long i, j;
	const float* __restrict gx{ gX.get() };
	const float* __restrict gy{ gY.get() };
	      float* __restrict a { A.get() };
	      float* __restrict b { B.get() };
	      float* __restrict c { C.get() };

	for (j = 0; j < sizeY; j++)
	{
		for (i = 0; i < sizeX; i++)
		{
			const A_long idx = j * sizeX + i;
			const float gx_ij = gx[idx];
			const float gy_ij = gy[idx];
			a[idx] = gx_ij * gx_ij;
			b[idx] = gy_ij * gy_ij;
			c[idx] = gx_ij * gy_ij;
		}
	}
	return;
}

inline void gaussian_kernel
(
	std::unique_ptr<float[]>& gKernel,
	const A_long& radius, 
	const float&  sigma
) noexcept
{
	float* __restrict kernel{ gKernel.get() };

	for (A_long i = -radius; i <= radius; i++)
		kernel[i + radius] = std::exp(-(std::pow(static_cast<float>(i) / sigma, 2.f) / 2.f));

	return;
}


void convolution
(
	std::unique_ptr<float[]>& gImIn,
	std::unique_ptr<float[]>& gKernel,
	const float& sigma,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& radius,
	std::unique_ptr<float[]>& gImOut
) noexcept
{
	auto imOutXSmartPtr = std::make_unique<float[]>(sizeX * sizeY);
	auto __restrict imOutX = imOutXSmartPtr.get();
	auto __restrict imOut  = gImOut.get();
	const float* __restrict imIn   = gImIn.get();
	const float* __restrict kernel = gKernel.get();

	A_long x, y, i, i0;

	/* convolution in X */
	for (y = 0; y < sizeY; y++)
	{
		for (x = 0; x < sizeX; x++)
		{
			i0 = y * sizeX + x;
			float sumV = 0.f;
			float sumK = 0.f;

			for (i = -radius; i <= radius; i++)
			{
				if ((x + i < 0) || (x + i > sizeX - 1))
					continue;
				const float valK = kernel[i + radius];
				sumV += imIn[i0 + i] * valK;
				sumK += valK;
			}

			imOutX[i0] = sumV / sumK;
		}
	}

	/* convolution in Y	*/
	for (y = 0; y < sizeY; y++)
	{
		for (x = 0; x < sizeX; x++)
		{
			i0 = y * sizeX + x;
			float sumV = 0.f;
			float sumK = 0.f;

			for (i = -radius; i <= radius; i++)
			{
				if ((y + i < 0) || (y + i > sizeY - 1))
					continue;
				const float valK = kernel[i + radius];
				sumV += imOutX[i0 + i * sizeX] * valK;
				sumK += valK;
			}

			imOut[i0] = sumV / sumK;
		}
	}
	return;
}


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
) noexcept
{
	/* create kernel */
	const A_long radius = static_cast<A_long>(std::ceil(2.f * sigma));
	const A_long kernelSize = 2 * radius + 1;
	auto gKernel = std::make_unique<float[]>(kernelSize);

#ifdef _DEBUG
	/* native buffer pointers forn DBG purpose only */
	auto dbgKernel = gKernel.get();
	auto dbg_A = A.get();
	auto dbg_B = B.get();
	auto dbg_C = C.get();
	auto dbg_A_reg = A_reg.get();
	auto dbg_B_reg = B_reg.get();
	auto dbg_C_reg = C_reg.get();
#endif

	gaussian_kernel (gKernel, radius, sigma);

	convolution (A, gKernel, sigma, sizeX, sizeY, radius, A_reg);
	convolution (B, gKernel, sigma, sizeX, sizeY, radius, B_reg);
	convolution (C, gKernel, sigma, sizeX, sizeY, radius, C_reg);

	return;
}


void diagonalize_structure_tensors
(
	const std::unique_ptr<float[]>& A,
	const std::unique_ptr<float[]>& B,
	const std::unique_ptr<float[]>& C,
	A_long sizeX,
	A_long sizeY,
	std::unique_ptr<float[]>& Lambda1,
	std::unique_ptr<float[]>& Lambda2,
	std::unique_ptr<float[]>& Eigvect2_x,
	std::unique_ptr<float[]>& Eigvect2_y,
	std::unique_ptr<float[]>& Anisotropy
) noexcept
{
	const float* __restrict a{ A.get() };
	const float* __restrict b{ B.get() };
	const float* __restrict c{ C.get() };
	float* __restrict lambda1{ Lambda1.get() };
	float* __restrict lambda2{ Lambda2.get() };
	float* __restrict eigvect2_x{ Eigvect2_x.get() };
	float* __restrict eigvect2_y{ Eigvect2_y.get() };
	float* __restrict anisotropy{ Anisotropy.get() };

	for (A_long j = 0; j < sizeY; j++)
	{
		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long idx{ j * sizeX + i };

			const float delta = (a[idx] - b[idx])*(a[idx] - b[idx]) + 4.f * c[idx] * c[idx];
			const float sqrtDelta = std::sqrt(delta);
			const float trace = a[idx] + b[idx];

			lambda1[idx] = 0.5f * (trace + sqrtDelta);
			lambda2[idx] = 0.5f * (trace - sqrtDelta);

			anisotropy[idx] = (0.f == trace ? 0.f : 1.f - 2.f * lambda2[idx] / trace);

			const float x1 = 2.f * c[idx];
			const float x2 = b[idx] - a[idx] - std::sqrt(delta);
			const float norm_eig_vect = std::sqrt(x1 * x1 + x2 * x2);

			if (norm_eig_vect > 0.f)
			{
				eigvect2_x[idx] = x1 / norm_eig_vect;
				eigvect2_y[idx] = x2 / norm_eig_vect;
			}
			else
			{
				eigvect2_x[idx] = eigvect2_y[idx] = 0.f;
			}
		}
	}

	return;
}


void pixel_list
(
	A_long* __restrict row_list,
	A_long* __restrict col_list,
	const A_long i_min1,
	const A_long i_min2,
	const A_long i_max1,
	const A_long i_max2,
	const A_long j_min1,
	const A_long j_min2,
	const A_long j_max1,
	const A_long j_max2
) noexcept
{
	A_long k, l, index = 0;

	for (k = i_min1; k <= i_max1; k++)
	{
		for (l = j_min1; l <= j_max1; l++)
		{
			row_list[index] = k;
			col_list[index] = l;
			index++;
		}
	}

	for (k = i_min2; k <= i_max2; k++)
	{
		for (l = j_min2; l <= j_max2; l++)
		{
			row_list[index] = k;
			col_list[index] = l;
			index++;
		}
	}

	return;
}


inline float test_adjacency
(
	const float& x1, 
	const float& y1,
	const float& v1x,
	const float& v1y,
	const float& x2,
	const float& y2,
	const float& v2x,
	const float& v2y,
	const float& thresh_cocirc, 
	const float& thresh_cone
) noexcept
{
	float resp = 0.f;
	float cocircularity = 0.f;

	// First test conic constraint
	float dx = x2 - x1;
	float dy = y2 - y1;
	const float n = std::sqrt(dx * dx + dy * dy);
	if (n > 0.f)
	{
		dx = dx / n;
		dy = dy / n;
	}
	auto const v1_dot_d = v1x * dx + v1y * dy;
	auto const conic_constraint = (std::abs(v1_dot_d) >= thresh_cone);

	// Then test co-circularity if conic constraint is fullfilled
	if (conic_constraint)
	{
		const float v0x = 2.f * v1_dot_d * dx - v1x;
		const float v0y = 2.f * v1_dot_d * dy - v1y;
		cocircularity = std::abs(v0x * v2x + v0y * v2y);
		resp = static_cast<float>(cocircularity >= thresh_cocirc);
	}
	return resp;
}

#ifdef _DEBUG
volatile A_long dbgCnt = 0;
volatile A_long respCnt = 0;
volatile A_long loopIntCnt = 0;
#endif

void compute_adjacency_matrix
(
	SparseMatrix<float>& S,
	std::unique_ptr<float[]>& Eigvect2_x,
	std::unique_ptr<float[]>& Eigvect2_y,
	A_long p,
	const A_long sizeX,
	const A_long sizeY,
	const float thresh_cocirc,
	const float thresh_cone
) noexcept
{
	float* __restrict eigvect2_x{ Eigvect2_x.get() };
	float* __restrict eigvect2_y{ Eigvect2_y.get() };
	A_long i_min1 = 0, i_max1 = 0, i_min2 = 0, i_max2 = 0;
	A_long j_min1 = 0, j_max1 = 0, j_min2 = 0, j_max2 = 0;
	A_long x1 = 0, y1 = 0, v1x = 0, v1y = 0;

	for (A_long i = 0; i < sizeY; i++)
	{
		i_min1 = i + 1;
		i_max1 = ((i + p < sizeY) ? i + p : sizeY - 1);
		i_min2 = ((i > p) ? i - p : 0);
		i_max2 = i;

		for (A_long j = 0; j < sizeX; j++)
		{
			j_min1 = j;
			j_max1 = ((j + p < sizeX) ? j + p : sizeX - 1);
			j_min2 = j + 1;
			j_max2 = j_max1;

			x1 = j;
			y1 = i;
			v1x = eigvect2_x[i * sizeX + j];
			v1y = eigvect2_y[i * sizeX + j];

			auto const h1 = ((i_max1 - i_min1 + 1 > 0) ? i_max1 - i_min1 + 1 : 0);
			auto const w1 = ((j_max1 - j_min1 + 1 > 0) ? j_max1 - j_min1 + 1 : 0);
			auto const h2 = ((i_max2 - i_min2 + 1 > 0) ? i_max2 - i_min2 + 1 : 0);
			auto const w2 = ((j_max2 - j_min2 + 1 > 0) ? j_max2 - j_min2 + 1 : 0);

			auto const n_pixels = h1 * w1 + h2 * w2;

			auto Row_list = std::make_unique<A_long []>(n_pixels);
			auto Col_list = std::make_unique<A_long []>(n_pixels);

			if (Row_list && Col_list)
			{
				auto __restrict row_list = Row_list.get();
				auto __restrict col_list = Col_list.get();

				pixel_list (row_list, col_list, i_min1, i_min2, i_max1, i_max2, j_min1, j_min2, j_max1, j_max2);

				for (A_long index = 0; index < n_pixels; index++)
				{
					auto l = col_list[index];
					auto k = row_list[index];
					auto x2 = l;
					auto y2 = k;
					auto v2x = eigvect2_x[k * sizeX + l];
					auto v2y = eigvect2_y[k * sizeX + l];

#ifdef _DEBUG
					loopIntCnt++;
#endif

					const float resp = test_adjacency (x1, y1, v1x, v1y, x2, y2, v2x, v2y, thresh_cocirc, thresh_cone);

					if (resp > FastCompute::RECIPROC_EXP)
					{
#ifdef _DEBUG
						respCnt++;
#endif
						S(i * sizeX + j, k * sizeX + l) = resp;
					}
				} /* for (A_long index = 0; index < n_pixels; index++) */

			} /* if (Row_list && Col_list) */

		} /* for (j = 0; j < sizeX; j++) */
#ifdef _DEBUG
			dbgCnt++;
#endif
	}

	return;
}


inline A_long count_sparse_matrix_non_zeros
(
	SparseMatrix<float>& S,
	const A_long& n_col
) noexcept
{
	A_long n_non_zeros = 0;
	for (A_long j = 0; j < n_col; j++)
	{
		auto col_j = S.get_column(j);
		for (std::map<A_long, float>::const_iterator it = col_j.begin(); it != col_j.end(); ++it)
			n_non_zeros++;
	}
	return n_non_zeros;
}


A_long count_sparse_matrix_non_zeros
(
	std::unique_ptr<SparseMatrix<float>>& S,
	const A_long& n_col
) noexcept
{
	return count_sparse_matrix_non_zeros(*S, n_col);
}


inline void sparse_matrix_to_arrays_impl 
(
	SparseMatrix<float>& S,
	A_long* __restrict I,
	A_long* __restrict J,
	float* __restrict W,
	const A_long& n_col
) noexcept
{
	A_long index = -1;

	for (A_long j = 0; j < n_col; j++)
	{
		auto col_j = S.get_column(j);
		for (auto it = col_j.begin(); it != col_j.end(); ++it)
		{
			index++;
			J[index] = j;
			I[index] = it->first;
			W[index] = it->second;
		}
	}
	return;
}

void sparse_matrix_to_arrays
(
	std::unique_ptr<SparseMatrix<float>>& S,
	std::unique_ptr<A_long []>& I,
	std::unique_ptr<A_long []>& J,
	std::unique_ptr<float  []>& W,
	A_long n_col
) noexcept
{
	A_long* __restrict i{ I.get() };
	A_long* __restrict j{ J.get() };
	float*  __restrict w{ W.get() };

	return sparse_matrix_to_arrays_impl (*S, i, j, w, n_col);
}



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
) noexcept
{
	/* TODO: needs understand how to significantly decrease memory usage into this ALGO */
	auto const& frameSize = CreateAlignment (width * height, CACHE_LINE);
	auto gX		= std::make_unique<float []>(frameSize);
	auto gY		= std::make_unique<float []>(frameSize);
	auto a		= std::make_unique<float []>(frameSize);
	auto b		= std::make_unique<float []>(frameSize);
	auto c		= std::make_unique<float []>(frameSize);
	auto a_reg	= std::make_unique<float []>(frameSize);
	auto b_reg	= std::make_unique<float []>(frameSize);
	auto c_reg	= std::make_unique<float []>(frameSize);
	auto lambda1= std::make_unique<float []>(frameSize);
	auto lambda2= std::make_unique<float []>(frameSize);

	auto eigvect2_x = std::make_unique<float []>(frameSize);
	auto eigvect2_y = std::make_unique<float []>(frameSize);
	auto anisotropy = std::make_unique<float []>(frameSize);

	bool retResult = false;

	if (gX && gY && a && b && c && a_reg && b_reg && c_reg && lambda1 && lambda2 && eigvect2_x && eigvect2_y && anisotropy)
	{
#ifdef _DEBUG
		/* native buffer pointers forn DBG purpose only */
		auto dbgGx = gX.get();
		auto dbgGy = gY.get();
		auto dbgA = a.get();
		auto dbgB = b.get();
		auto dbgC = c.get();
		auto dbgAreg = a_reg.get();
		auto dbgBreg = b_reg.get();
		auto dbgCreg = c_reg.get();
		auto dbgLam1 = lambda1.get();
		auto dbgLam2 = lambda2.get();
		auto dbgEigx = eigvect2_x.get();
		auto dbgEigy = eigvect2_y.get();
#endif

		linear_gradient_gray (im, gX, gY, width, height, pitch);
		structure_tensors0 (gX, gY, width, height, a, b, c);
		smooth_structure_tensors (a, b, c, sigma, width, height, a_reg, b_reg, c_reg);

		diagonalize_structure_tensors (a_reg, b_reg, c_reg, width, height, lambda1, lambda2, eigvect2_x, eigvect2_y, anisotropy);

		auto __restrict Anisotropy{ anisotropy.get() };
		for (A_long j = 0; j < height; j++)
		{
			float* pSrc = im_anisotropy + j * pitch;
			for (A_long i = 0; i < width; i++)
				pSrc[i] = Anisotropy[i] * 255.f;
		}

		compute_adjacency_matrix (S, eigvect2_x, eigvect2_y, p, width, height, coCirc, coCone);
		retResult = true;
	}
	return retResult;
}


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
) noexcept
{
	return bw_image2cocircularity_graph_impl (im, *S, im_anisotropy, width, height, pitch, sigma, coCirc, coCone, p);
}


bool erode_max_plus_symmetric
(
	const float* __restrict imIn,
	float*  __restrict imOut,
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const A_long& nLines,
	const float& norm
) noexcept
{
	bool change = false;

	__VECTOR_ALIGNED__
	for (A_long l = 0; l < nLines; l++)
	{
		const auto i{ I[l] };
		const auto j{ J[l] };
		const float w = norm * std::log (W[l]);
		if (imOut[j] + w > imIn[i])
		{
			imOut[j] = imIn[i] - w;
			change = true;
		}
		if (imOut[i] + w > imIn[j])
		{
			imOut[i] = imIn[j] - w;
			change = true;
		}
	}

	return change;
}


int erode_max_plus_symmetric_iterated
(
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const float*  __restrict imIn,
	float*  __restrict imOut[],
	const A_long& k,
	const A_long& n_lines,
	float** pOut,
	const float& normalizer
) noexcept
{
	A_long iteration = 0;
	bool changed = true;

	const float* __restrict imgSrc = imIn;
	float* __restrict imgDst = imOut[0];

	while (iteration < k && true == changed)
	{
		changed = erode_max_plus_symmetric (imgSrc, imgDst, I, J, W, n_lines, normalizer);

		if (nullptr != pOut) { *pOut = imgDst; }

		iteration++;
		imgSrc = imgDst;
		imgDst = imOut[iteration & 0x1];
	} /* while (iteration < k && true == changed) */

	return (true == changed ? k : iteration - 1);
}


bool dilate_max_plus_symmetric
(
	const float* __restrict imIn,
	float*  __restrict imOut,
	const A_long* __restrict I,
	const A_long* __restrict J,
	const float*  __restrict W,
	const A_long& nLines,
	const float& norm
) noexcept
{
	bool change = false;

	__VECTOR_ALIGNED__
	for (A_long l = 0; l < nLines; l++)
	{
		const auto& i = I[l];
		const auto& j = J[l];
		
		const float w = norm * std::log(W[l]);
		if (imOut[i] < imIn[j] + w)
		{
			imOut[i] = imIn[j] + w;
			change = true;
		}
		if (imOut[j] < imIn[i] + w)
		{
			imOut[j] = imIn[i] + w;
			change = true;
		}
	}
	return change;
}


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
	const float& normalizer
) noexcept
{
	A_long iteration = 0;
	bool changed = true;

	const float* __restrict imgSrc = imIn;
	float* __restrict imgDst = imOut[0];

	while (iteration < k && true == changed)
	{
		changed = dilate_max_plus_symmetric (imgSrc, imgDst, I, J, W, n_lines, normalizer);

		if (nullptr != pOut) { *pOut = imgDst; }

		iteration++;
		imgSrc = imgDst;
		imgDst = imOut[iteration & 0x1];
	} /* while (iteration < k && true == changed) */

	return (true == changed ? k : iteration - 1);
}


inline A_long morpho_open_impl
(
	      float*  __restrict imIn,
	      float*  __restrict imOut,
	const float*  __restrict Weights,
	const A_long* __restrict I,
	const A_long* __restrict J,
	const A_long& it,
	const A_long& nonZeros,
	const A_long& sizeX,
	const A_long& sizeY,
	const float&  normalizer
) noexcept
{
	const A_long frameSize = sizeX * sizeY;
	auto imProc1 = std::make_unique<float []>(frameSize);
	auto imProc2 = std::make_unique<float []>(frameSize);
	float* pOut{ nullptr };

	if (imProc1 && imProc2)
	{
		float* __restrict im_proc[2]{ imProc1.get(), imProc2.get() };
		const A_long kMax = erode_max_plus_symmetric_iterated (I, J, Weights, imIn, im_proc, it, nonZeros, &pOut, normalizer);
		const size_t memSize = frameSize * sizeof(float);

		if (kMax == it)
		{
			memcpy (imIn, pOut, memSize);
			dilate_max_plus_symmetric_iterated (I, J, Weights, imIn, im_proc, it, nonZeros, &pOut, normalizer);
		} /* if (kMax == it) */

		memcpy (imOut, pOut, memSize);
	} /* if (imProc1 && imProc2) */

	return 0;
}


A_long morpho_open
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
	float  normalizer
) noexcept
{
	const float*  __restrict W{ w.get() };
	const A_long* __restrict I{ i.get() };
	const A_long* __restrict J{ j.get() };

	return morpho_open_impl (imIn, imOut, W, J, I, it, nonZeros, sizeX, sizeY, normalizer);
}