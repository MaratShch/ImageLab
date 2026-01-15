#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_BASE_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_BASE_CLASS__

#include <array>
#include <vector>
#include <cmath>
#include "Common.hpp"
#include "IPainter.hpp"
#include "PainterPalette/ArtPointillismPalette.hpp"
#include "AlgoColorConversion.hpp" 

class PainterBase : public IPainter {
protected:
    // Standard float arrays (No enforced alignment)
    // We keep size 32 to allow AVX to read 8-float chunks safely
    // even if the actual palette is only 10 or 24 colors.
    float m_L[32];
    float m_a[32];
    float m_b[32];
    int32_t m_actual_size;

    StrokeShape m_shape;
    ColorMode   m_mode;
    bool        m_use_flow;

public:
    // Default constructor/destructor
    virtual ~PainterBase() = default;

    void SetupContext(RenderContext& ctx) const override final
    {
        ctx.pal_L = m_L;
        ctx.pal_a = m_a;
        ctx.pal_b = m_b;
        ctx.palette_size = m_actual_size;
        ctx.shape = m_shape;
        ctx.color_mode = m_mode;
        ctx.use_orientation = m_use_flow;
    }

protected:
    template <typename T, size_t N>
    void Initialize(const std::array<PEntry<T>, N>& source, StrokeShape shape, ColorMode mode, bool use_flow)
    {    
        m_shape = shape; m_mode = mode; m_use_flow = use_flow;
        
        // Clamp to max buffer size (32)
        size_t safe_count = (N > 32) ? 32 : N;
        m_actual_size = (int32_t)safe_count;

        for (size_t i = 0; i < safe_count; ++i) {
            float r, g, b, L, a_val, b_val;
            
            if (std::is_integral<T>::value)
            {
                r = static_cast<float>(source[i].r) / 255.0f;
                g = static_cast<float>(source[i].g) / 255.0f;
                b = static_cast<float>(source[i].b) / 255.0f;
            }
            else
            {
                r = static_cast<float>(source[i].r);
                g = static_cast<float>(source[i].g);
                b = static_cast<float>(source[i].b);
            }

            Convert_RGB_to_Lab_Scalar(r, g, b, L, a_val, b_val);

            m_L[i] = L;
            m_a[i] = a_val;
            m_b[i] = b_val;
        }

        // Padding with "Infinity" for safety 
        // (Ensures unused AVX lanes don't accidentally match)
        for (size_t i = safe_count; i < 32; ++i) {
            m_L[i] = 10000.0f;
            m_a[i] = 10000.0f;
            m_b[i] = 10000.0f;
        }
    }
};

#endif