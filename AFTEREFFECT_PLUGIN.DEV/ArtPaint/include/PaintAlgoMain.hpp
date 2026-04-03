#pragma once

#include <cstdint>
#include "PaintMemHandler.hpp"
#include "PaintAlgoContols.hpp"
#include "PaintAlgoAVX2.hpp"
#include "PaintMorpologyKernels.hpp"


void PaintAlgorithmMain (const MemHandler& memHndl, const AlgoControls& algoCtrl, A_long width, A_long height);
