#include "PainterSeurat.hpp"

SeuratPainter* painterSeurat = nullptr;

std::array<IPainter*, UnderlyingType(ArtPointillismPainter::ART_POINTILLISM_PAINTER_TOTAL_NUMBER)> painterInterface{};


IPainter* GetPainterRegistry (ArtPointillismPainter)
{
	//TODO....
	return nullptr;
}