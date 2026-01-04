#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismMatisse.hpp"

// =========================================================
// HENRI MATISSE (Fauvist / Divisionist Phase)
// Technique: Energetic, expressive strokes, non-naturalistic color.
// =========================================================
class MatissePainter final : public PainterBase {
public:
    MatissePainter()
	{
        // Shape: Oriented Ellipse (Dashes/Flowing strokes)
        // Mode: Expressive (Boosts saturation before matching)
        // Flow: True (Follows image gradients/contours)
        Initialize
		(
			Matisse_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_ELLIPSE,
			ColorMode::Expressive,
			true
		);
    }

	const char* GetShortName () const override { return "Matisse"; }
    const char* GetName() const override { return "Henri Matisse"; }
};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_CLASS__