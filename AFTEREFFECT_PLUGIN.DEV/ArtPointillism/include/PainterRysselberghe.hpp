#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismRysselberghe.hpp"

// =========================================================
// THEO VAN RYSSELBERGHE (Belgian Neo-Impressionist)
// Technique: Precise, jewel-like square touches.
// =========================================================
class RysselberghePainter final : public PainterBase {
public:
    RysselberghePainter()
	{
        // Shape: Square (Classic Divisionist Mosaic)
        // Mode: Scientific
        // Flow: False
        Initialize
		(
			Rysselberghe_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_SQUARE,
			ColorMode::Scientific,
			false
		);
    }

    virtual ~RysselberghePainter() = default;

	const char* GetShortName () const override { return "Rysselberghe"; }
    const char* GetName() const override { return "Theo van Rysselberghe"; }
};



#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_CLASS__