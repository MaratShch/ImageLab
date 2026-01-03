#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_INTERFACE__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_INTERFACE__

#include <algorithm>
#include <cstdint>
#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismEnums.hpp"


struct RenderContext
{
    const fCIELabPix* palette_buffer;
    int32_t         palette_size;
    StrokeShape     shape;
    ColorMode       color_mode;
    bool            use_orientation;
};

constexpr size_t RenderContextSize = sizeof(RenderContext);

class IPainter 
{
public:
    virtual ~IPainter() = default;

    // The Contract: Every painter must be able to configure the rendering context.
    virtual void SetupContext (RenderContext& ctx) const = 0;

    // Optional: Useful for debug/logging
    virtual const char* GetShortName() const = 0;
    virtual const char* GetName() const = 0;


protected:

private:

};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_TECHNICS_INTERFACE__