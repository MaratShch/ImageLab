#pragma once

#include "NoiseCleanPrototypes.hpp"

constexpr char strName[] = "Noise Clean";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rNoise Clean plugin.";
constexpr int NoiseClean_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int NoiseClean_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int NoiseClean_VersionSub = 0;
#ifdef _DEBUG
constexpr int NoiseClean_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int NoiseClean_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int NoiseClean_VersionBuild = 1;


/* FUNCTION PROTOTYPES */
bool LoadMemoryInterfaceProvider(int32_t appId, int32_t major, int32_t minor = 0) noexcept;
int32_t GetMemoryBlock(int32_t size, int32_t align, void** pMem) noexcept;
void FreeMemoryBlock(int32_t id) noexcept;

void gaussian_weights (A_long filterRadius, float gMesh[][cBilateralWindowMax]) noexcept;
void gaussian_weights (A_long filterRadius, float* __restrict gMesh) noexcept;

PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;
