#ifndef __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "AlgoMemHandler.hpp"
#include "ArtPointillismEnums.hpp"
#include "AlgoLumaManipulation.hpp"
#include "AlgoDotEngine.hpp"
#include "AlgoJFA.hpp"
#include "AlgoArtisticsRendering.hpp"
#include "AssembleFinalImage.hpp"


void ArtPointillismAlgorithmExec
(
	const MemHandler& algoMemHandler,
	const PontillismControls& algoControls,
	const int32_t sizeX,
	const int32_t sizeY
);


#endif // __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__