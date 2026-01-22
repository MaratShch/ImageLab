#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARRO_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARRO_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismPissarro.hpp"

// =========================================================
// CAMILLE PISSARRO (Neo-Impressionist Period 1886-1890)
// Technique: Fine, dusty dots, very similar to Seurat.
// =========================================================
class PissarroPainter final : public PainterBase
{
public:
    PissarroPainter()
	{
        // Shape: Circle (Cluster effect)
        // Mode: Scientific (Strict decomposition)
        // Flow: False
        Initialize
		(
			Pissarro_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_CIRCLE,
			ColorMode::Scientific,
			false
		);
    }

    virtual ~PissarroPainter() = default;

	const char* GetShortName () const override { return "Pissarro"; }
    const char* GetName() const override { return "Camille Pissarro"; }
};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARO_CLASS__