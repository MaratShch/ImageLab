#include "Kernels.hpp"
#include "Convolution.hpp"
#include <atomic>

static std::atomic<uint32_t> b{ 0u };

template <typename T>
CACHE_ALIGN IAbsrtactKernel<T>* factoryKernels[KERNEL_CONV_SIZE]{};

template <typename T>
IAbsrtactKernel<T>* GetKernel (uint32_t idx)
{
	return ((idx < KERNEL_CONV_SIZE) ? factoryKernels<T>[idx] : nullptr);
}

template <typename T>
static inline void SetKernel(IAbsrtactKernel<T>* iKernel, uint32_t idx)
{
	 if (idx < KERNEL_CONV_SIZE)
		 factoryKernels<T>[idx] = iKernel;
}

template <typename T>
void InitKernels(void)
{
	factoryKernels<T>[0]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Sharp3x3<T>);
	factoryKernels<T>[1]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Sharp5x5<T>);
	factoryKernels<T>[2]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Blur3x3<T>);
	factoryKernels<T>[3]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Blur5x5<T>);
	factoryKernels<T>[4]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Sharpen3x3Factor<T>);
	factoryKernels<T>[5]  = reinterpret_cast<IAbsrtactKernel<T>*>(new IntenceSharpen3x3<T>);
	factoryKernels<T>[6]  = reinterpret_cast<IAbsrtactKernel<T>*>(new EdgeDetection<T>);
	factoryKernels<T>[7]  = reinterpret_cast<IAbsrtactKernel<T>*>(new Edge45Degrees<T>);
	factoryKernels<T>[8]  = reinterpret_cast<IAbsrtactKernel<T>*>(new EdgeHorizontal<T>);
	factoryKernels<T>[9]  = reinterpret_cast<IAbsrtactKernel<T>*>(new EdgeVertical<T>);
	factoryKernels<T>[10] = reinterpret_cast<IAbsrtactKernel<T>*>(new Emboss<T>);
	factoryKernels<T>[11] = reinterpret_cast<IAbsrtactKernel<T>*>(new IntenseEmboss<T>);
	factoryKernels<T>[12] = reinterpret_cast<IAbsrtactKernel<T>*>(new Soften3x3<T>);
	factoryKernels<T>[13] = reinterpret_cast<IAbsrtactKernel<T>*>(new Soften5x5<T>);
	factoryKernels<T>[14] = reinterpret_cast<IAbsrtactKernel<T>*>(new Gaussian3x3<T>);
	factoryKernels<T>[15] = reinterpret_cast<IAbsrtactKernel<T>*>(new Gaussian5x5<T>);
	factoryKernels<T>[16] = reinterpret_cast<IAbsrtactKernel<T>*>(new Laplacian3x3<T>);
	factoryKernels<T>[17] = reinterpret_cast<IAbsrtactKernel<T>*>(new Laplacian5x5<T>);
	factoryKernels<T>[18] = reinterpret_cast<IAbsrtactKernel<T>*>(new MotionBlur9x9<T>);
	factoryKernels<T>[19] = reinterpret_cast<IAbsrtactKernel<T>*>(new MotionBlurL2R9x9<T>);
	factoryKernels<T>[20] = reinterpret_cast<IAbsrtactKernel<T>*>(new MotionBlurR2L9x9<T>);
	factoryKernels<T>[21] = reinterpret_cast<IAbsrtactKernel<T>*>(new HighPass<T>);
	return;
}

template <typename T>
void FreeKernels(void)
{
	for (uint32_t i = 0; i < KERNEL_CONV_SIZE; i++)
	{
		IAbsrtactKernel<T>* iKernel = GetKernel<T>(i);
		if (nullptr != iKernel)
		{
			delete iKernel;
			iKernel = nullptr;
			SetKernel<T>(iKernel, i);
		}
	}
	return;
}


void InitKernelsFactory(void)
{
	if (0u == b)
	{
		b = 1u;
		InitKernels<float>();
		InitKernels<int32_t>();
	}
	return;
}


void FreeKernelsFactory(void)
{
	FreeKernels<float>();
	FreeKernels<int32_t>();
	b = 0u;
	return;
}