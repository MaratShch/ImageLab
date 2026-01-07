#include "PainterFactory.hpp"


static std::array<IPainter*, UnderlyingType(ArtPointillismPainter::ART_POINTILLISM_PAINTER_TOTAL_NUMBER)> painterInterface{};

bool CreatePaintersEngine(void)
{
	painterInterface[0] = reinterpret_cast<IPainter*>(new SeuratPainter);
	painterInterface[1] = reinterpret_cast<IPainter*>(new SignacPainter);
	painterInterface[2] = reinterpret_cast<IPainter*>(new CrossPainter);
	painterInterface[3] = reinterpret_cast<IPainter*>(new PissarroPainter);
	painterInterface[4] = reinterpret_cast<IPainter*>(new VanGoghPainter);
	painterInterface[5] = reinterpret_cast<IPainter*>(new MatissePainter);
	painterInterface[6] = reinterpret_cast<IPainter*>(new RysselberghePainter);
	painterInterface[7] = reinterpret_cast<IPainter*>(new LucePainter);
	
	return true;
}

void DeletePaintersEngine (void)
{
	for (auto& painter : painterInterface)
	{
		if (nullptr != painter)
		{
			delete painter;
			painter = nullptr;
		}
	}
}


IPainter* GetPainterRegistry (ArtPointillismPainter painterID)
{
	const int32_t id = UnderlyingType(painterID);
	if (id >= 0 && id < painterInterface.size())
		return painterInterface[id];
		
	return nullptr;
}