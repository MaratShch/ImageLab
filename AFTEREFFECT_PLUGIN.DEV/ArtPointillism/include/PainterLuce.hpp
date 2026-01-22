#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismLuce.hpp"

// =========================================================
// MAXIMILIEN LUCE (Neo-Impressionist)
// Technique: Energetic divisionism, often distinct blocks of color.
// =========================================================
class LucePainter final : public PainterBase
{
public:
    LucePainter()
	{
        // Shape: Square
        // Mode: Scientific
        // Flow: False
        Initialize
		(
			Luce_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_SQUARE,
			ColorMode::Scientific,
			false
		);
    }

    virtual ~LucePainter() = default;

    const char* GetShortName() const override { return "Luce"; }
    const char* GetName() const override { return "Maximilien Luce"; }
};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_CLASS__