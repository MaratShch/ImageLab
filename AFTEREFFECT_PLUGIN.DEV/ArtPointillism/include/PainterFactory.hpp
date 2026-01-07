#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_FACTORY_CLASS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_FACTORY_CLASS__


#include "ArtPointillismEnums.hpp"
#include "PainterCross.hpp"
#include "PainterLuce.hpp"
#include "PainterMatisse.hpp"
#include "PainterPissarro.hpp"
#include "PainterRysselberghe.hpp"
#include "PainterSeurat.hpp"
#include "PainterSignac.hpp"
#include "PainterVanGogh.hpp"

bool CreatePaintersEngine(void);
void DeletePaintersEngine(void);

IPainter* GetPainterRegistry (ArtPointillismPainter);

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_FACTORY_CLASS__