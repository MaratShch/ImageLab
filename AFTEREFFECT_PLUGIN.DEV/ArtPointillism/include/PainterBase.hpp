#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_BASE_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_BASE_CLASS__

#include <array>
#include <vector>
#include "IPainter.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismColorConvert.hpp"
#include "PainterPalette/ArtPointillismPalette.hpp"

class PainterBase : public IPainter
{
protected:
    // Internal Data (Heap allocated vector, created ONCE at startup)
    std::vector<fCIELabPix> m_palette_lab;
    
    // Configuration
    StrokeShape m_shape;
    ColorMode   m_mode;
    bool        m_use_flow;

public:
    // Implementation of the Interface
    void SetupContext (RenderContext& ctx) const override final
	{
        ctx.palette_buffer  = m_palette_lab.data();
        ctx.palette_size    = static_cast<int32_t>(m_palette_lab.size());
        ctx.shape           = m_shape;
        ctx.color_mode      = m_mode;
        ctx.use_orientation = m_use_flow;
    }

protected:
    // Protected Init function used by derived constructors
    template <typename T, size_t N>
	void Initialize
	(
        const std::array<PEntry<T>, N>& source_palette, 
        StrokeShape shape, 
        ColorMode mode,
        bool use_flow
    ) noexcept
	{
        m_shape = shape;
        m_mode = mode;
        m_use_flow = use_flow;

        m_palette_lab.reserve(N);
        
        for (const auto& entry : source_palette)
        {
            fRGB paletteVal;

            // Handle uint8 vs float input
            if (std::is_integral<T>::value)
            {
                paletteVal.R = F32(entry.r);
                paletteVal.G = F32(entry.g);
                paletteVal.B = F32(entry.b);
            } else
            {
                paletteVal.R = static_cast<float>(entry.r);
                paletteVal.G = static_cast<float>(entry.g);
                paletteVal.B = static_cast<float>(entry.b);
            }

            // PERFORM CONVERSION HERE
            const fCIELabPix CIELabValue = Xyz2CieLab(Rgb2Xyz(paletteVal));    
            m_palette_lab.push_back(CIELabValue);
        }
    }
};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_BASE_CLASS__