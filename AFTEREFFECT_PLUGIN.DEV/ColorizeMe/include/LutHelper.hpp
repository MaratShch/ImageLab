#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include "Common.hpp"
#include "CubeLUT.h"

typedef std::vector<LutObjHndl> LutHelper;

using LutHndl = LutHelper*;
using LutIdx = int32_t;

constexpr size_t defaultLutHelperSize = 128;
constexpr size_t LutIdxSize = sizeof(LutIdx);

void InitLutHelper(const size_t& capacity = defaultLutHelperSize);
void DisposeAllLUTs (void);
LutIdx addToLut(const std::string& lutName);
LutObjHndl getLut (const LutIdx& idx);
void removeLut  (const LutIdx& idx);
