#include "AlgoControls.hpp"


AlgoControls getAlgoControlsDefault (void)
{
	AlgoControls algoCtrl{};
	
	algoCtrl.denoise_amount = 1.0f;
	algoCtrl.luma_strength = 1.0f;
	algoCtrl.chroma_strength = 1.0f;

	algoCtrl.detail_preservation = 0.0f; 
	
	algoCtrl.match_sensitivity = 1.0f; 
	
	algoCtrl.search_radius = 10; 
	
	algoCtrl.stride = 1; 
	
	algoCtrl.low_freq_mult = 1.0f;
	algoCtrl.high_freq_mult = 1.0f;
	
	algoCtrl.block_size = 8;

	algoCtrl.noise_curve_a = 25.0f; 
	algoCtrl.noise_curve_b = 0.0f;
	
	return algoCtrl;
}