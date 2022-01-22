#pragma once

#include "CommonAdobeAE.hpp"
#include <cfloat>

constexpr char strName[] = "Color Band Select";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rColor Band Select plugin.";
constexpr int ColorBandSelect_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorBandSelect_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorBandSelect_VersionSub   = 0;
#ifdef _DEBUG
constexpr int ColorBandSelect_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorBandSelect_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorBandSelect_VersionBuild = 1;
