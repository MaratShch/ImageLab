#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

constexpr int circular_size = 3;
constexpr int hist_size_H = 360;
constexpr int hist_size_S = 208;
constexpr int hist_size_I = 256;
constexpr unsigned int max_buf_idx = 4u;

constexpr int pointilismBrashSize = 11;
using tPointilismBrush = float[pointilismBrashSize][pointilismBrashSize];

const tPointilismBrush& getPointilismBrush1(size_t&);
const tPointilismBrush& getPointilismBrush2(size_t&);
const tPointilismBrush& getPointilismBrush3(size_t&);
const tPointilismBrush& getPointilismBrush4(size_t&);