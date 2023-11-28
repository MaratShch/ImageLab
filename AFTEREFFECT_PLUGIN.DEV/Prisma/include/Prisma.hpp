#ifndef __IMAGE_LAB_PRISMA_VIDEO_FILTER__
#define __IMAGE_LAB_PRISMA_VIDEO_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Prisma";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rPrisma plugin.";
constexpr int PrismaVideo_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int PrismaVideo_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int PrismaVideo_VersionSub = 0;
#ifdef _DEBUG
constexpr int PrismaVideo_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int PrismaVideo_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int PrismaVideo_VersionBuild = 1;


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


#endif /* __IMAGE_LAB_PRISMA_VIDEO_FILTER__ */
