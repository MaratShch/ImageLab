#ifndef __IMAGELAB2_ALGORITHM_MAIN_HPP__
#define __IMAGELAB2_ALGORITHM_MAIN_HPP__

// Declaration ONLY. AlgoControls and MemHandler are YOUR types and are
// defined in your own headers - this file must never redefine them.

#include <cstdint>
#include "AlgoControl.hpp"          // YOUR AlgoControls
#include "AlgoMemHandler.hpp"           // YOUR MemHandler  <-- adjust this include
                                    //     to whatever your header is called
#include "cct_interface.hpp"        // AlgoCCT::CctHandle
#include "super_pixel.hpp"          // SuperPixel<>, CctDuv<>
#include "AlgoReference.hpp"        // AlgoWB::WbReference

void Algorithm_Main (AlgoCCT::CctHandle<double>& cctHdnl,
                     const SuperPixel<double>&   superPixel,
                     const MemHandler&           memHandler,
                     const int32_t               sizeX,
                     const int32_t               sizeY,
                     const AlgoControls&         params,
                     CctDuv<double>&             cct_duv,
                     const AlgoWB::WbReference*  reference = nullptr) noexcept;

#endif // __IMAGELAB2_ALGORITHM_MAIN_HPP__
