#pragma once

#include "CommonAdobeAE.hpp"

PF_Err PR_ImageEq_Linear_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_BGRA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;


PF_Err PR_ImageEq_Linear_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Linear_RGB_444_10u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Sigmoid_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err PR_ImageEq_Sigmoid_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;