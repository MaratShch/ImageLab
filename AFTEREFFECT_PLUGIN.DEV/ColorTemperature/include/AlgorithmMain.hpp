#pragma once

#include <cstdint>
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"
#include "super_pixel.hpp"
#include "color_utils.hpp"
#include "cct_uv_to_xyz.hpp"   // AlgoCCT::uv_to_XYZ (reverse path)

#include "Algo2Rgb2XYZ.hpp"

void Algorithm_Main
(
    AlgoCCT::CctHandle<double>& cctHdnl,    
    const SuperPixel<double>& superPixel, // Previously computed SuperPixel	
    const MemHandler&     memHandler, 	// contains linearized input and output RGB buffers, and buffers for intermediate processing/compute
    const int32_t         sizeX,	// horizontal linearized image size in pixels	
    const int32_t         sizeY,	// vertical linearized image size in pixels
    const AlgoControls&   params,	// Algorithm Control parameters
    CctDuv<double>&       cct_duv   // Computed CCT and Duv/Tint values    
) ;
