#include "color_utils.hpp"


// super-pixel RGB (double) -> (u,v) -> CCT/Duv
void superpixel_to_cct
(
	const SuperPixel<double>& sp,
    AlgoCCT::CctHandle<double>& cctHndl,
    const double* rgb_to_xyz,   // working-space matrix (row-major 3x3)
    eCOLOR_OBSERVER observer,
    double& cct,
	double& duv
)
{
    // 1. RGB -> XYZ
    const double X = rgb_to_xyz[0]*sp.r + rgb_to_xyz[1]*sp.g + rgb_to_xyz[2]*sp.b;
    const double Y = rgb_to_xyz[3]*sp.r + rgb_to_xyz[4]*sp.g + rgb_to_xyz[5]*sp.b;
    const double Z = rgb_to_xyz[6]*sp.r + rgb_to_xyz[7]*sp.g + rgb_to_xyz[8]*sp.b;

    // 2. XYZ -> CIE 1960 u,v
    const double den = X + 15.0*Y + 3.0*Z;
    const double u = 4.0*X / den;
    const double v = 6.0*Y / den;

    // 3+4. nearest-entry + parabolic refine (coarse+fine, both inside)
    const std::pair<double, double> cct_duv = cctHndl.ComputeCct({u, v}, observer);   // -> cct, duv  (wraps refine())
    
    cct = cct_duv.first;
    duv = cct_duv.second;
    
    return;
}