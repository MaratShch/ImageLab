#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImagePaintUtils.hpp"


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
	const A_long shortSizeX = sizeX - 1;
	const A_long shortSizeY = sizeY - 1;

	__VECTOR_ALIGNED__
	for (j = 0; j < sizeY; j++)
	{
		const float* __restrict pSrc = im + j * pitch;
		float* __restrict gx = gX.get() + j * sizeX;
		gx[j] = pSrc[1] - pSrc[0];

		for (i = 1; i < shortSizeX; i++)
			gx[i] = 0.5f * (pSrc[i + 1] - pSrc[i - 1]);

		gx[i] = pSrc[i] - pSrc[i - 1];
	}

	{
		const float* __restrict pSrcLine0 = im;
		const float* __restrict pSrcLine1 = im + pitch;
		float* __restrict gy = gY.get();

		__VECTOR_ALIGNED__
		for (i = 0; i < sizeX; i++)
			gy[i] = pSrcLine1[i] - pSrcLine0[i];
	}

	__VECTOR_ALIGNED__
	for (j = 1; j < shortSizeY; j++)
	{
		const float* __restrict pSrcLine0 = im + (j - 1) * pitch;
		const float* __restrict pSrcLine2 = im + (j + 1) * pitch;
		float* __restrict gy = gY.get() + j * sizeX;

		for (i = 0; i < sizeX; i++)
			gy[i] = 0.5f * (pSrcLine2[i] - pSrcLine0[i]);
	}

	{
		const float* __restrict pSrcLine0 = im + (j - 1) * pitch;
		const float* __restrict pSrcLine1 = im + j * pitch;
		float* __restrict gy = gY.get();

		__VECTOR_ALIGNED__
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

	__VECTOR_ALIGNED__
	for (j = 0; j < sizeY; j++)
	{
		for (i = 0; i < sizeX; i++)
		{
			const A_long idx = j * sizeX + i;
			const float& gx_ij = gx[idx];
			const float& gy_ij = gy[idx];
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
	float* kernel = gKernel.get();

	for (A_long i = -radius; i <= radius; i++)
		kernel[i + radius] = FastCompute::Exp(-(FastCompute::Pow(static_cast<float>(i) / sigma, 2.f) / 2.f));

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
	auto const& frameSize = CreateAlignment (sizeX * sizeY, CACHE_LINE);
	auto imOutXSmartPtr = std::make_unique<float[]>(frameSize);

	float* __restrict imOutX = imOutXSmartPtr.get();
	float* __restrict imOut  = gImOut.get();
	const float* __restrict imIn   = gImIn.get();
	const float* __restrict kernel = gKernel.get();

	A_long x, y, i, i0;

	/* convolution in X */
	__VECTOR_ALIGNED__
	for (y = 0; y < sizeX; y++)
	{
		for (x = 0; x < sizeY; x++)
		{
			i0 = y * sizeX + x;
			float sumV{ 0.f };
			float sumK{ 0.f };

			for (i = -radius; i <= radius; i++)
			{
				if ((x + i < 0) || (x + i > sizeX - 1))
					continue;
				const float& valK{ kernel[i + radius] };
				sumV += imIn[i0 + i] * valK;
				sumK += valK;
			}

			imOutX[i0] = sumV / sumK;
		}
	}

	/* convolution in Y	*/
	__VECTOR_ALIGNED__
	for (y = 0; y < sizeY; y++)
	{
		for (x = 0; x < sizeX; x++)
		{
			i0 = y * sizeX + x;
			float sumV{ 0.f };
			float sumK{ 0.f };

			for (i = -radius; i <= radius; i++)
			{
				if ((y + i < 0) || (y + i > sizeX - 1))
					continue;
				const float& valK{ kernel[i + radius] };
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
	const A_long radius = static_cast<int>(std::ceil(2.f * sigma));
	const A_long kernelSize = 2 * radius + 1;
	auto gKernel = std::make_unique<float[]>(kernelSize);

	gaussian_kernel (gKernel, radius, sigma);

	convolution (A, gKernel, sigma, sizeY, sizeX, radius, A_reg);
	convolution (B, gKernel, sigma, sizeY, sizeX, radius, B_reg);
	convolution (C, gKernel, sigma, sizeY, sizeX, radius, C_reg);

	return;
}


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

	linear_gradient_gray (im, gX, gY, width, height, pitch);

	structure_tensors0 (gX, gY, width, height, a, b, c);

	smooth_structure_tensors (a, b, c, sigma, width, height, a_reg, b_reg, c_reg);

	return true;
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
	return bw_image2cocircularity_graph (im, *S, im_anisotropy, width, height, pitch, sigma, coCirc, coCone, p);
}

