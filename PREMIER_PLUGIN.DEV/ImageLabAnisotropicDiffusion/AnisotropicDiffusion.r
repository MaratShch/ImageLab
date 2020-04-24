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
#define plugInName		"Perona-Malik denoise"
#define plugCategory 	"ImageLab"

// This name should not be localized or updated
#define plugInMatchName	"ImageLab_Perona_Malik_denoise"


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
			4,					
			plugInMatchName
		},

		ANIM_ParamAtom 
		{
			0,					// Property count - zero-based count
			"Dispersion",		// Parameter name
			1,					// Parameter number - one-based count
			ANIM_DT_SHORT,		// Data type
			ANIM_UI_SLIDER,		// UI Type
			0x40040000,			// valid_min 2
			0x0,					
			0x40280000,			// valid_max 12
			0x0,					
			0x40040000,			// ui_min 2
			0x0,					
			0x40280000,			// ui_max 12
			0x0,				
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
			"Time Step",		// Parameter name
			2,					// Parameter number - one-based count
			ANIM_DT_SHORT,		// Data type
			ANIM_UI_SLIDER,		// UI Type
			0x40020000,			// valid_min 1
			0x0,					
			0x40240000,			// valid_max 10
			0x0,					
			0x40020000,			// ui_min 1
			0x0,					
			0x40240000,			// ui_max 10
			0x0,				
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
			2,					// Property count - zero-based count
			"Noise level",		// Parameter name
			3,					// Parameter number - one-based count
			ANIM_DT_SHORT,		// Data type
			ANIM_UI_SLIDER,		// UI Type
			0x40020000,			// valid_min 0
			0x0,					
			0x40240000,			// valid_max 10
			0x0,					
			0x40020000,			// ui_min 0
			0x0,					
			0x40240000,			// ui_max 10
			0x0,				
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
			3,					// Property count - zero-based count
			"Advanced",			// Parameter name
			4,					// Parameter number - one-based count
			13,					// Data type - ANIM_DT_BOOLEAN
			8,					// UI Type - ANIM_UI_CHECKBOX
			0x0,
			0x0,				// valid_min (0) - not used by Premiere
			0x0,
			0x0,				// valid_max (0) - not used by Premiere
			0x0,
			0x0,				// ui_min (0)
			0x0,				
			0x0,				// ui_max (0)
#if (PiPLVerMajor >= 2) && (PiPLVerMinor >= 3)
			dontScaleUIRange,
#endif
			animateParam,		// Set/don't set this to indicate if the param should be animated
			dontRestrictBounds,	// Rest of these aren't used by Premiere
			spaceIsAbsolute,
			resIndependent,
			1					// Bytes size of the param data
		},

	}
};

