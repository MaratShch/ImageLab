#include "ArtPointillismAlgo.hpp"

void ArtPointillismAlgorithmExec
(
	const MemHandler& algoMemHandler,
	const PontillismControls& algoControls,
	const int32_t sizeX,
	const int32_t sizeY
)
{
	const int32_t pointOutMaxSize = static_cast<int32_t>(static_cast<float>(sizeX * sizeY) * max_dots_per_pixel * 1.15f + 0.5f);

	///////////////////////////////////////////////////////////////////////////////////////
	// ALGO STAGE 1: Luma channel manipulations
	///////////////////////////////////////////////////////////////////////////////////////
	
	// 1.2 invert Luma channel
	CIELab_LumaInvert (algoMemHandler.L, algoMemHandler.Luma1, sizeX, sizeY);
		
	// 1.3 Luma edge detection
	LumaEdgeDetection (algoMemHandler.Luma1, algoMemHandler.Luma2, sizeX, sizeY);
		
	// 1.4 MinMax density map
	const int32_t pixel_count = sizeX * sizeY;
	MixAndNormalizeDensity (algoMemHandler.Luma1, algoMemHandler.Luma2, algoMemHandler.DencityMap, pixel_count, algoControls.EdgeSensitivity); 
		
	///////////////////////////////////////////////////////////////////////////////////////
	// ALGO STAGE 2: Linear Content-Aware Seeding
	///////////////////////////////////////////////////////////////////////////////////////
	
	// 2.1 Dot Budgeting
	int32_t dotsNumb = CalculateTargetDotHighCount (sizeX, sizeY, algoControls.DotDencity);
	// SAFETY: Never ask for more dots than we have memory for.
	const int32_t dots = std::min(dotsNumb, pointOutMaxSize - 1);
		
	// 2.2 Calculate number of dots for each node
	int32_t actualPoints = 0;
		Run_Seeding (algoMemHandler.DencityMap, sizeX, sizeY, algoControls.DotDencity, algoControls.RandomSeed,
					algoMemHandler.NodePool, algoMemHandler.NodeElemNumber, algoMemHandler.PointOut, dots, &actualPoints);

	///////////////////////////////////////////////////////////////////////////////////////
	// ALGO STAGE 3: Geometric Refinement
	///////////////////////////////////////////////////////////////////////////////////////
	constexpr int32_t relaxation_iterations = 4;
	JFAPixel* voronoi_map = // !!!! actually returned pointer to algoMemHandler.JfaBuffer- Ping/Pong
		Dot_Refinement (algoMemHandler.PointOut, actualPoints, algoMemHandler.DencityMap, sizeX, sizeY, relaxation_iterations, 
				    algoMemHandler.JfaBufferPing, algoMemHandler.JfaBufferPong, algoMemHandler.AccumX, algoMemHandler.AccumY, algoMemHandler.AccumW);

	///////////////////////////////////////////////////////////////////////////////////////
	// ALGO STAGE 4: Artistic Rendering
	///////////////////////////////////////////////////////////////////////////////////////
    ArtisticRendering (algoMemHandler.PointOut, actualPoints, voronoi_map, algoMemHandler.L, algoMemHandler.ab, 
	                   algoMemHandler.DencityMap, sizeX, sizeY, algoControls, algoMemHandler.Scratch, algoMemHandler.CanvasLab);
						   
	return;				   
}
