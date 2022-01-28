#pragma once

#include "CommonAdobeAE.hpp"
#include <cfloat>
#include <mutex>
#include "SE_Interface.hpp"

constexpr char strName[] = "Color Band Select";
constexpr char strCopyright[] = "\n2019-2022. ImageLab2 Copyright(c).\rColor Band Select plugin.";
constexpr int MorphologyFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int MorphologyFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int MorphologyFilter_VersionSub   = 0;
#ifdef _DEBUG
constexpr int MorphologyFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int MorphologyFilter_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int MorphologyFilter_VersionBuild = 1;


typedef struct strSeData
{
	uint32_t bValid;
	SE_Interface* IstructElem;

	strSeData::strSeData()
	{
		bValid = false;
		IstructElem = nullptr;
	}
	strSeData::~strSeData()
	{
		delete IstructElem;
		IstructElem = nullptr;
		bValid = false;
	}
}strSeData;
constexpr A_long strSeDataSize = static_cast<A_long>(sizeof(strSeData));


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