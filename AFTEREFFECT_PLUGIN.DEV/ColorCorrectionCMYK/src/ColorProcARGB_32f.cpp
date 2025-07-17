#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "RGB2CMYK.hpp"

PF_Err aeProcessImage_ARGB_4444_32f_CMYK
(
    PF_InData*		in_data,
    PF_OutData*		out_data,
    PF_ParamDef*	params[],
    PF_LayerDef*	output,
    float           add_c,
    float           add_m,
    float           add_y,
    float           add_k
) noexcept
{
    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*       __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    auto const height = output->height;
    auto const width  = output->width;

    float C, M, Y, K;
    float newC, newM, newY, newK;
    float newR, newG, newB;

    constexpr float reciproc100 = 1.0f / 100.0f;

#ifdef _DEBUG	
    C = M = Y = K = 0.f;
    newC = newM = newY = newK = 0.f;
    newR = newG = newB = 0.f;
#endif

    for (auto j = 0; j < height; j++)
    {
        auto const src_line_idx = j * src_pitch;
        auto const dst_line_idx = j * dst_pitch;

        __VECTOR_ALIGNED__
        for (auto i = 0; i < width; i++)
        {
            PF_Pixel_ARGB_32f const& srcPixel = localSrc[src_line_idx + i];

            float const& B = srcPixel.B;
            float const& G = srcPixel.G;
            float const& R = srcPixel.R;
            auto  const& A = srcPixel.A;

            rgb_to_cmyk(R, G, B, C, M, Y, K);

            newC = CLAMP_VALUE(C + add_c * reciproc100, 0.f, 1.0f);
            newM = CLAMP_VALUE(M + add_m * reciproc100, 0.f, 1.0f);
            newY = CLAMP_VALUE(Y + add_y * reciproc100, 0.F, 1.0f);
            newK = CLAMP_VALUE(K + add_k * reciproc100, 0.f, 1.0f);

            cmyk_to_rgb(newC, newM, newY, newK, newR, newG, newB);

            /* put to output buffer updated value */
            localDst[dst_line_idx + i].B = newB;
            localDst[dst_line_idx + i].G = newG;
            localDst[dst_line_idx + i].R = newR;
            localDst[dst_line_idx + i].A = A;

         } /* for (i = 0; i < width; i++) */

    } /* for (j = 0; j < height; j++) */

    return PF_Err_NONE;
}



PF_Err aeProcessImage_ARGB_4444_32f_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b
) noexcept
{
    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*       __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    auto const height = output->height;
    auto const width  = output->width;

    constexpr float reciproc = 1.0f / static_cast<float>(coarse_max_level);
    float const addR = add_r * reciproc;
    float const addG = add_g * reciproc;
    float const addB = add_b * reciproc;

    for (auto j = 0; j < height; j++)
    {
        auto const& src_line_idx = j * src_pitch;
        auto const& dst_line_idx = j * dst_pitch;

        __VECTOR_ALIGNED__
        for (auto i = 0; i < width; i++)
        {
            auto const src_pix_idx = src_line_idx + i;
            auto const dst_pix_idx = dst_line_idx + i;

            /* put to output buffer updated value */
            localDst[dst_pix_idx].B = CLAMP_VALUE(localSrc[src_pix_idx].B + addB, f32_value_black, f32_value_white);
            localDst[dst_pix_idx].G = CLAMP_VALUE(localSrc[src_pix_idx].G + addG, f32_value_black, f32_value_white);
            localDst[dst_pix_idx].R = CLAMP_VALUE(localSrc[src_pix_idx].R + addR, f32_value_black, f32_value_white);
            localDst[dst_pix_idx].A = localSrc[src_pix_idx].A;

        } /* for (i = 0; i < width; i++) */

    } /* for (j = 0; j < height; j++) */

    return PF_Err_NONE;
}