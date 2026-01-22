#include "PainterFactory.hpp"


static std::array<IPainter*, UnderlyingType(ArtPointillismPainter::ART_POINTILLISM_PAINTER_TOTAL_NUMBER)> painterInterface{};

bool CreatePaintersEngine(void)
{
    if (nullptr == painterInterface[0])
	    painterInterface[0] = reinterpret_cast<IPainter*>(new SeuratPainter);
    if (nullptr == painterInterface[1])
        painterInterface[1] = reinterpret_cast<IPainter*>(new SignacPainter);
    if (nullptr == painterInterface[2])
        painterInterface[2] = reinterpret_cast<IPainter*>(new CrossPainter);
    if (nullptr == painterInterface[3])
        painterInterface[3] = reinterpret_cast<IPainter*>(new PissarroPainter);
    if (nullptr == painterInterface[4])
        painterInterface[4] = reinterpret_cast<IPainter*>(new VanGoghPainter);
    if (nullptr == painterInterface[5])
        painterInterface[5] = reinterpret_cast<IPainter*>(new MatissePainter);
    if (nullptr == painterInterface[6])
        painterInterface[6] = reinterpret_cast<IPainter*>(new RysselberghePainter);
    if (nullptr == painterInterface[7])
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