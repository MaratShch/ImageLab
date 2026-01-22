#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismSignac.hpp"

class SignacPainter final : public PainterBase
{
public:
    SignacPainter()
	{
        // Seurat_u8 is your global constexpr array
        Initialize
		(
			Signac_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_SQUARE,
			ColorMode::Scientific,
			false
		);
    }
    
    virtual ~SignacPainter() = default;
    
    const char* GetShortName () const override { return "Signac"; }
    const char* GetName() const override { return "Paul Signac"; }
};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_CLASS__