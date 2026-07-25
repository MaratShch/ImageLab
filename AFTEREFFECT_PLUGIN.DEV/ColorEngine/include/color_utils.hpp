#pragma once

#include <cstdint>
#include <type_traits>
#include "Common.hpp"
#include "ColorTransformMatrix.hpp"
#include "super_pixel.hpp"
#include "cct_interface.hpp"

void superpixel_to_cct
(
	const SuperPixel<double>& sp,
    AlgoCCT::CctHandle<double>& cctHndl,
    const double* rgb_to_xyz,   // working-space matrix (row-major 3x3)
    eCOLOR_OBSERVER observer,
    double& cct,
	double& duv
);