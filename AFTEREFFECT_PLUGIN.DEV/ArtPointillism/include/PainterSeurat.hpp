#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_CLASS__

#include "ArtPointillismEnums.hpp"
#include "PainterBase.hpp"
#include "PainterPalette/ArtPointillismSeurat.hpp"

class SeuratPainter final : public PainterBase
{
public:
    SeuratPainter()
	{
        // Seurat_u8 is your global constexpr array
        Initialize
		(
			Seurat_f32,
			StrokeShape::ART_POINTILLISM_SHAPE_CIRCLE,
			ColorMode::Scientific,
			false
		);
    }

    virtual ~SeuratPainter() = default;
    
	const char* GetShortName () const override { return "Seurat"; }
    const char* GetName() const override { return "Georges Seurat"; }
};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_CLASS__