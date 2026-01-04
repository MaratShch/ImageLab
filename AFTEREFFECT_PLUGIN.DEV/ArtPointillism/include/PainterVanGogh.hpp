#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_VANGOGH_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_VANGOGH_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismVanGogh.hpp"

class VanGoghPainter final : public PainterBase
{
public:
    VanGoghPainter()
	{
        // Seurat_u8 is your global constexpr array
        Initialize
		(
			VanGogh_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_ELLIPSE,
			ColorMode::Expressive,
			true
		);
    }
    
	const char* GetShortName () const override { return "Van Gogh"; }
    const char* GetName() const override { return "Vincent van Gogh"; }
};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_VANGOGH_CLASS__