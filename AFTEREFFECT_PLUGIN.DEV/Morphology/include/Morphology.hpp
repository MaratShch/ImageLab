#pragma once

#include "CommonAdobeAE.hpp"
#include <cfloat>
#include <mutex>
#include "SE_Interface.hpp"
#include "SequenceData.hpp"

constexpr char strName[] = "Morphology Filter";
constexpr char strCopyright[] = "\n2019-2022. ImageLab2 Copyright(c).\rMorphology Filter plugin.";
constexpr int MorphologyFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int MorphologyFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int MorphologyFilter_VersionSub   = 0;
#ifdef _DEBUG
constexpr int MorphologyFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int MorphologyFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int MorphologyFilter_VersionBuild = 1;


constexpr A_long strSeDataSize{ static_cast<A_long>(sizeof(std::uint64_t)) };
constexpr A_long maxSeLineSize{ 9 };
constexpr A_long maxSeElemNumber{ maxSeLineSize * maxSeLineSize };

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



inline const SE_Interface* getStructuredElemInterface(const PF_OutData* __restrict out_data) noexcept
{
	std::uint64_t seIdx{ INVALID_INTERFACE };

	/* get Structured Element Object */
	const std::uint64_t* seData{ reinterpret_cast<uint64_t*>(GET_OBJ_FROM_HNDL(out_data->sequence_data)) };
	if (nullptr == seData)
		return nullptr;

	if (INVALID_INTERFACE == (seIdx = *seData))
		return nullptr;

	return DataStore::getObject(seIdx);
}


