#include <vector>
#include "fft.hpp"
#include "dft.hpp"



void FourierTransform::mixed_radix_fft_1D (const float* in, float* out, int32_t size)
{
	const std::vector<int32_t> prime_vector = FourierTransform::prime (size);
//	std::cout << "Prime factors for size = " << size << std::endl;
//	for (const auto& elem : prime_vector)
//		std::cout << elem << " ";
//	std::cout << std::endl;

    // 1. Check if all factors are supported by your efficient kernels
    bool is_fast_path = true;
    for (int32_t f : prime_vector)
	{
        if (f != 16 && f != 8 && f != 7 && f != 5 && f != 4 && f != 3 && f != 2)
		{
            is_fast_path = false;
            break;
        }
    }

    // 2. Dispatch
    if (is_fast_path)
    {
        // Perfect! Your Iterative Engine handles this.
        FFT_MixedRadix_Iterative (in, out, size, prime_vector);
    }
    else
    {
        // Fallback for primes/unsupported sizes (e.g. 11, 13, 1009)
        if (size < 128)
        {
            // Assumes you kept the DFT_Canonical implementation
            FourierTransform::dft_1D(in, out, size);
        } else
        {
            // Assumes you kept the FFT_CZT implementation
            FourierTransform::fft_czt(in, out, size);
        }
    }
    
    return;
}	


void FourierTransform::mixed_radix_fft_1D (const double* in, double* out, int32_t size)
{
	const std::vector<int32_t> prime_vector = FourierTransform::prime (size);
//	std::cout << "Prime factors for size = " << size << std::endl;
//	for (const auto& elem : prime_vector)
//		std::cout << elem << " ";
//	std::cout << std::endl;

    // 1. Check if all factors are supported by your efficient kernels
    bool is_fast_path = true;
    for (int32_t f : prime_vector)
	{
        if (f != 16 && f != 8 && f != 7 && f != 5 && f != 4 && f != 3 && f != 2)
		{
            is_fast_path = false;
            break;
        }
    }

    // 2. Dispatch
    if (is_fast_path)
    {
        // Perfect! Your Iterative Engine handles this.
        FFT_MixedRadix_Iterative (in, out, size, prime_vector);
    }
    else
    {
        // Fallback for primes/unsupported sizes (e.g. 11, 13, 1009)
        if (size < 128)
        {
            // Assumes you kept the DFT_Canonical implementation
            FourierTransform::dft_1D(in, out, size);
        } else
        {
            // Assumes you kept the FFT_CZT implementation
            FourierTransform::fft_czt(in, out, size);
        }
    }
    
    return;
}	


// ============================================================================
// INVERSE FFT 1D (Wrapper)
// ============================================================================
// Computes IFFT using the Conjugate property: IFFT(x) = conj(FFT(conj(x))) / N
// This reuses the highly optimized Forward engine without code duplication.
// ----------------------------------------------------------------------------
void FourierTransform::mixed_radix_ifft_1D (const float* __restrict in, float* __restrict out, int32_t size)
{
    // 1. Pre-Process: Copy Input to Output AND Conjugate
    // We perform the copy here manually to flip the sign of the imaginary part.
    // in[k] = a + jb  ->  out[k] = a - jb
    
    for (int32_t i = 0; i < size; ++i)
    {
        out[2*i]     =  in[2*i];      // Real part (unchanged)
        out[2*i + 1] = -in[2*i + 1];  // Imag part (flipped)
    }

    // 2. Execute Forward FFT
    // We perform this IN-PLACE on the 'out' buffer.
    // Since 'out' now holds conj(input), the result will be unscaled conj(output).
    mixed_radix_fft_1D(out, out, size);

    // 3. Post-Process: Conjugate again AND Normalize
    // We need to flip the imaginary sign back, and divide by N.
    
    const float normalization_factor = 1.0f / static_cast<float>(size);
    
    for (int32_t i = 0; i < size; ++i)
    {
        // Apply scaling
        out[2*i]     *= normalization_factor; 
        
        // Apply scaling AND flip sign
        // Current val is (a + jb). We want (a - jb) * scale.
        out[2*i + 1] *= -normalization_factor; 
    }
    
    return;
}


void FourierTransform::mixed_radix_ifft_1D  (const double* __restrict in, double* __restrict out, int32_t size)
{
    // 1. Pre-Process: Copy Input to Output AND Conjugate
    // We perform the copy here manually to flip the sign of the imaginary part.
    // in[k] = a + jb  ->  out[k] = a - jb
    
    for (int32_t i = 0; i < size; ++i)
    {
        out[2*i]     =  in[2*i];      // Real part (unchanged)
        out[2*i + 1] = -in[2*i + 1];  // Imag part (flipped)
    }

    // 2. Execute Forward FFT
    // We perform this IN-PLACE on the 'out' buffer.
    // Since 'out' now holds conj(input), the result will be unscaled conj(output).
    mixed_radix_fft_1D (out, out, size);

    // 3. Post-Process: Conjugate again AND Normalize
    // We need to flip the imaginary sign back, and divide by N.
    
    const double normalization_factor = 1.0 / static_cast<double>(size);
    
    for (int32_t i = 0; i < size; ++i)
    {
        // Apply scaling
        out[2*i]     *= normalization_factor; 
        
        // Apply scaling AND flip sign
        // Current val is (a + jb). We want (a - jb) * scale.
        out[2*i + 1] *= -normalization_factor; 
    }
    
    return;
}
// 
//	Input buffer should be coming in interleaved format: RE, IM, RE, IM .../
//	Same interleaved format produced for output buffer
//
//void FourierTransform::mixed_radix_fft_2D (const float* in, float* out, int32_t sizeX, int32_t sizeY)
//{
//	mixed_radix_fft_1D (in, out, sizeX);
//	mixed_radix_fft_1D (in, out, sizeY);
//	
//	return;
//}
