#pragma once

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "Convolution";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rImage convolution with differents kernels plugin.";
constexpr int Convolution_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int Convolution_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int Convolution_VersionSub   = 0;
#ifdef _DEBUG
constexpr int Convolution_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int Convolution_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int Convolution_VersionBuild = 1;

constexpr char KernelType[] = "Kernel Type";

constexpr char strKernels[] =	"Sharp 3x3|"
								"Sharp 5x5|"
	                            "Blur 3x3|"
	                            "Blur 5x5|" 
								"Sharpen 3x3 Factor|"
								"Intense Sharpen 3x3|"
								"Edge Detection|"
								"Edge 45 Degrees|"
								"Edge Horizontal|"
								"Edge Vertical|"
								"Emboss|"
								"Intense Emboss|"
								"Soften 3x3|"
								"Soften 5x5|"
								"Gaussian 3x3|"
								"Gaussian 5x5|"
								"Laplasian 3x3|"
								"Laplasian 5x5|"
								"Motion Blur 9x9|"
								"Motion Blur -> 9x9|"
								"Motion Blur <- 9x9|"
	                            "Custom Kernel";

enum {
	CONVOLUTION_INPUT,
	KERNEL_CHECKBOX,
	CONVLOVE_NUM_PARAMS
};

typedef enum {
	KERNEL_CONV_SHARP_3x3 = 0,
	KERNEL_CONV_SHARP_5x5,
	KERNEL_CONV_BLUR_3x3,
	KERNEL_CONV_BLUR_5x5,
	KERNEL_CONV_SHARPEN_3x3,
	KERNEL_CONV_INTENSE_SHARPEN,
	KERNEL_CONV_EDGE_DETECTION,
	KERNEL_CONV_EDGE_45_DEGREES,
	KERNEL_CONV_EDGE_HORIZONTAL,
	KERNEL_CONV_EDGE_VERTICAL,
	KERNEL_CONV_EMBOSS,
	KERNEL_CONV_INTENSE_EMBOSS,
	KERNEL_CONV_SOFTEN_3x3,
	KERNEL_CONV_SOFTEN_5x5,
	KERNEL_CONV_GAUSSIAN_3x3,
	KERNEL_CONV_GAUSSIAN_5x5,
	KERNEL_CONV_LAPLASIAN_3x3,
	KERNEL_CONV_LAPLASIAN_5x5,
	KERNEL_CONV_MOTON_BLUR_9x9,
	KERNEL_CONV_BLUR_LEFTRIGHT_9x9,
	KERNEL_CONV_BLUR_RIGHTLEFT_9x9,
	KERNEL_CONV_CUSTOM_KERNEL,
	KERNEL_CONV_SIZE
} ILab2KernelType;

enum {
	KERNEL_CONV_DISK_ID = 1
};

typedef struct {
	PF_PixelFloat	color;
} prerender_stuff, *pre_render_stuffP, **pre_render_stuffH;


bool ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat& destinationPixelFormat = PrPixelFormat_Invalid
);

bool ProcessImgInAE
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
);