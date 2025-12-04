#include <vector>
#include "fft.hpp"
#include "dft.hpp"
#include "utils.hpp"


void FourierTransform::mixed_radix_fft_1D (const float* in, float* out, int32_t size) noexcept
{
	const std::vector<int32_t> prime_vector = FourierTransform::prime (size);

    // 1. Check if all factors are supported by your efficient kernels
    bool is_fast_path = true;
    for (const auto& f : prime_vector)
	{
        if (f > 9 && f != 16) // prime-function guarantee f >= 2
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
        if (size < dft_algo_threshold)
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


void FourierTransform::mixed_radix_fft_1D (const double* in, double* out, int32_t size) noexcept
{
	const std::vector<int32_t> prime_vector = FourierTransform::prime (size);

    // 1. Check if all factors are supported by your efficient kernels
    bool is_fast_path = true;
    for (const auto& f : prime_vector)
    {
        if (f > 9 && f != 16) // prime-function guarantee f >= 2
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
        if (size < dft_algo_threshold)
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
void FourierTransform::mixed_radix_ifft_1D (const float* RESTRICT in, float* RESTRICT out, int32_t size) noexcept
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


void FourierTransform::mixed_radix_ifft_1D  (const double* RESTRICT in, double* RESTRICT out, int32_t size) noexcept
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


void FourierTransform::mixed_radix_fft_2D (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, int32_t width, int32_t height) noexcept
{
    // Optimization: Read directly from 'in', write to 'out'.
    // This eliminates the initial memcpy entirely.
    // If (in == out), the 1D engine handles in-place safely.

    const int32_t double_width = 2 * width;
    for (int32_t row = 0; row < height; ++row)
    {
        // Offset in floats (2 per complex pixel)
        const int32_t offset = row * double_width;

        // Read from Input, Write to Output
        FourierTransform::mixed_radix_fft_1D (in + offset, out + offset, width);
    }

    // Input: 'out' (Row-transformed data)
    // Output: 'scratch' (Transposed data)
    // Dimensions change: [W x H] -> [H x W]

    utils_transpose_complex_2d(out, scratch, width, height);

    // We now operate on 'scratch'.
    // The "Rows" of scratch correspond to the "Columns" of the original image.
    // Length of these rows is 'height'. There are 'width' such rows.

    const int32_t double_height = 2 * height;
    for (int32_t col = 0; col < width; ++col)
    {
        // Offset in floats
        const int32_t offset = col * double_height;
        float* row_ptr = scratch + offset;

        // In-Place transform on the scratch buffer
        FourierTransform::mixed_radix_fft_1D (row_ptr, row_ptr, height);
    }

    // Input: 'scratch' (Fully transformed, but transposed)
    // Output: 'out' (Final Result)
    // Dimensions restore: [H x W] -> [W x H]

    utils_transpose_complex_2d(scratch, out, height, width);

    return;
}


void FourierTransform::mixed_radix_fft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept
{
    // Optimization: Read directly from 'in', write to 'out'.
    // This eliminates the initial memcpy entirely.
    // If (in == out), the 1D engine handles in-place safely.

    const int32_t double_width = 2 * width;
    for (int32_t row = 0; row < height; ++row)
    {
        // Offset in floats (2 per complex pixel)
        const int32_t offset = row * double_width;

        // Read from Input, Write to Output
        FourierTransform::mixed_radix_fft_1D(in + offset, out + offset, width);
    }

    // Input: 'out' (Row-transformed data)
    // Output: 'scratch' (Transposed data)
    // Dimensions change: [W x H] -> [H x W]

    utils_transpose_complex_2d(out, scratch, width, height);

    // We now operate on 'scratch'.
    // The "Rows" of scratch correspond to the "Columns" of the original image.
    // Length of these rows is 'height'. There are 'width' such rows.

    const int32_t double_height = 2 * height;
    for (int32_t col = 0; col < width; ++col)
    {
        // Offset in floats
        const int32_t offset = col * double_height;
        double* row_ptr = scratch + offset;

        // In-Place transform on the scratch buffer
        FourierTransform::mixed_radix_fft_1D (row_ptr, row_ptr, height);
    }

    // Input: 'scratch' (Fully transformed, but transposed)
    // Output: 'out' (Final Result)
    // Dimensions restore: [H x W] -> [W x H]

    utils_transpose_complex_2d(scratch, out, height, width);

    return;
}


void FourierTransform::mixed_radix_ifft_2D (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, int32_t width, int32_t height) noexcept
{
    // ========================================================================
    // PASS 1: IFFT ROWS
    // ========================================================================
    // Optimization: Read directly from 'in', write to 'out'.
    // The mixed_radix_ifft_1D function internally handles the "Copy + Conjugate"
    // step, so we can pass different pointers for source and destination here.

    const int32_t double_width = 2 * width;

    for (int32_t row = 0; row < height; ++row)
    {
        const int32_t offset = row * double_width;

        // Read from Input, Write to Output (Row by Row)
        FourierTransform::mixed_radix_ifft_1D(in + offset, out + offset, width);
    }

    // ========================================================================
    // PASS 2: TRANSPOSE (Rows -> Cols)
    // ========================================================================
    // Input: 'out' (Row-transformed data)
    // Output: 'scratch' (Transposed data)
    // Dimensions change: [W x H] -> [H x W]

    utils_transpose_complex_2d(out, scratch, width, height);

    // ========================================================================
    // PASS 3: IFFT "COLUMNS" (Processed as Rows)
    // ========================================================================
    // We now operate on 'scratch'.
    // The "Rows" of scratch correspond to the "Columns" of the original image.

    const int32_t double_height = 2 * height;

    for (int32_t col = 0; col < width; ++col)
    {
        const int32_t offset = col * double_height;
        float* row_ptr = scratch + offset;

        // In-Place transform on the scratch buffer
        FourierTransform::mixed_radix_ifft_1D(row_ptr, row_ptr, height);
    }

    // ========================================================================
    // PASS 4: TRANSPOSE BACK
    // ========================================================================
    // Input: 'scratch' (Fully transformed, but transposed)
    // Output: 'out' (Final Result)
    // Dimensions restore: [H x W] -> [W x H]

    utils_transpose_complex_2d(scratch, out, height, width);
    return;
}

void FourierTransform::mixed_radix_ifft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept
{
    // ========================================================================
    // PASS 1: IFFT ROWS
    // ========================================================================
    // Optimization: Read directly from 'in', write to 'out'.
    // The mixed_radix_ifft_1D function internally handles the "Copy + Conjugate"
    // step, so we can pass different pointers for source and destination here.

    const int32_t double_width = 2 * width;

    for (int32_t row = 0; row < height; ++row)
    {
        const int32_t offset = row * double_width;

        // Read from Input, Write to Output (Row by Row)
        FourierTransform::mixed_radix_ifft_1D(in + offset, out + offset, width);
    }

    // ========================================================================
    // PASS 2: TRANSPOSE (Rows -> Cols)
    // ========================================================================
    // Input: 'out' (Row-transformed data)
    // Output: 'scratch' (Transposed data)
    // Dimensions change: [W x H] -> [H x W]

    utils_transpose_complex_2d(out, scratch, width, height);

    // ========================================================================
    // PASS 3: IFFT "COLUMNS" (Processed as Rows)
    // ========================================================================
    // We now operate on 'scratch'.
    // The "Rows" of scratch correspond to the "Columns" of the original image.

    const int32_t double_height = 2 * height;

    for (int32_t col = 0; col < width; ++col)
    {
        const int32_t offset = col * double_height;
        double* row_ptr = scratch + offset;

        // In-Place transform on the scratch buffer
        FourierTransform::mixed_radix_ifft_1D(row_ptr, row_ptr, height);
    }

    // ========================================================================
    // PASS 4: TRANSPOSE BACK
    // ========================================================================
    // Input: 'scratch' (Fully transformed, but transposed)
    // Output: 'out' (Final Result)
    // Dimensions restore: [H x W] -> [W x H]

    utils_transpose_complex_2d(scratch, out, height, width);
    return;
}