#include "FastAriphmetics.hpp"
#include "ImageLabCUDA.hpp"
#include "AverageFilterGPU.hpp"


CUDA_KERNEL_CALL
void AverageFilter_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height
)
{
	return;
}