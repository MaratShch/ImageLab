#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismCross.hpp"

// =========================================================
// HENRI-EDMOND CROSS (Neo-Impressionist / Pre-Fauve)
// Technique: Distinct, blocky strokes (Tesserae).
// =========================================================
class CrossPainter final : public PainterBase
{
public:
    CrossPainter()
	{
        // Shape: Square (Mosaic effect)
        // Mode: Scientific (Optical mixing, though palette is vibrant)
        // Flow: False (Static grid)
        Initialize
		(
			Cross_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_SQUARE,
			ColorMode::Scientific,
			false
		);
    }

    const char* GetShortName() const override { return "Cross"; }
    const char* GetName() const override { return "Henri-Edmond Cross"; }
};



#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_CLASS__