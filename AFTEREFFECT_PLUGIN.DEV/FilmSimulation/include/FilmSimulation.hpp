#ifndef __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__
#define __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Film Simulation";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rFilm Simulation plugin.";
constexpr int FilmSimulation_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int FilmSimulation_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int FilmSimulation_VersionSub = 0;
#ifdef _DEBUG
constexpr int FilmSimulation_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int FilmSimulation_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int FilmSimulation_VersionBuild = 1;


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


#endif // __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__
