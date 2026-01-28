#pragma once

#include "CommonAdobeAE.hpp"
#include <cfloat>

constexpr char strName[] = "Sepia Color";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rSepia Color plugin.";
constexpr int SepiaColor_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int SepiaColor_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int SepiaColor_VersionSub   = 0;
#ifdef _DEBUG
constexpr int SepiaColor_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int SepiaColor_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int SepiaColor_VersionBuild = 1;

constexpr float value_black = 0.f;
constexpr float value_white = 1.0f - FLT_EPSILON;

constexpr int SEPIA_COLOR_INPUT = 0;