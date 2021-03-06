// Set PiPL version before including version header
#ifndef PiPLVerMajor
#define	PiPLVerMajor	2
#endif
#ifndef PiPLVerMinor
#define	PiPLVerMinor	3
#endif

#ifndef PRWIN_ENV
#define MAC_ENV
#include "PrSDKPiPLVer.h"
#include "PrSDKPiPL.r"
#endif

// The following two strings should be localized
#define plugInName		"VideoPuzzle"
#define plugCategory 	"ImageLab"

// This name should not be localized or updated
#define plugInMatchName	"ImageLab_VideoPuzzle"


resource 'PiPL' (16000)
{
	{	
		// The plug-in type
		Kind {PrVideoFilter},
		
		// The name as it will appear to the user
		Name {plugInName},
	
		// The internal name of this plug-in
		AE_Effect_Match_Name {plugInMatchName},

		// The folder containing the plug-in in the Effects Panel
		Category {plugCategory},

		// The version of the PiPL resource definition
		AE_PiPL_Version {PiPLVerMajor, PiPLVerMinor},
	
		// The ANIM properties describe the filter parameters, and also how the xplatform data is
		// stored in the project file. There is one FilterInfo property followed by n ParamAtoms

		ANIM_FilterInfo
		{
			0,
#if (PiPLVerMajor >= 2) && (PiPLVerMinor >= 3)
			notUnityPixelAspectRatio,
			anyPixelAspectRatio,
			reserved4False,
			reserved3False,
			reserved2False,
#endif
			reserved1False,		// Premiere doesn't use any of these flags, but AE does
			reserved0False,
			driveMe,
			doesntNeedDialog,	// needsDialog / doesntNeedDialog - Don't enable "Setup..."
			paramsNotPointer,
			paramsNotHandle,
			paramsNotMacHandle,
			dialogNotInRender,
			paramsNotInGlobals,
			bgAnimatable,
			fgAnimatable,
			geometric,
			noRandomness,
			2,					
			plugInMatchName
		},

		ANIM_ParamAtom 
		{
			0,					// Property count - zero-based count
			"Blocks Number",	// Parameter name
			1,					// Parameter number - one-based count
			ANIM_DT_SHORT,		// Data type
			ANIM_UI_SLIDER,		// UI Type
			0x40040000, // 2
			0x0, // valid_min (0.0)
			0x40300000, // 16
			0x0, // valid_max (16.0)
			0x40040000, // 2
			0x0, // ui_min (0.0)
			0x40300000, // 16
			0x0, // ui_max 
#if (PiPLVerMajor >= 2) && (PiPLVerMinor >= 3)
			dontScaleUIRange,
#endif
			animateParam,		// Set/don't set this to indicate if the param should be animated
			restrictBounds,	// Rest of these aren't used by Premiere
			spaceIsAbsolute,
			resIndependent,
			2					// Bytes size of the param data
		},

		ANIM_ParamAtom 
		{
			1,					// Property count - zero-based count
			"Tolerance",		// Parameter name
			0,					// Parameter number - one-based count
			ANIM_DT_SHORT,		// Data type
			ANIM_UI_SLIDER,		// UI Type
			0x40040000, // 2
			0x0, // valid_min
			0x40690000, // 200
			0x0, // valid_max
			0x40040000, // 2
			0x0, // ui_min
			0x40690000, // 200
			0x0, // ui_max
#if (PiPLVerMajor >= 2) && (PiPLVerMinor >= 3)
			dontScaleUIRange,
#endif
			animateParam,		// Set/don't set this to indicate if the param should be animated
			restrictBounds,	// Rest of these aren't used by Premiere
			spaceIsAbsolute,
			resIndependent,
			2					// Bytes size of the param data
		},

	}
};

