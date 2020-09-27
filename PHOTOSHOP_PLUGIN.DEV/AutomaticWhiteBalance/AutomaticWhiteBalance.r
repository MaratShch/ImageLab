#include "PIDefines.h"

#ifdef __PIMac__
	#include <Carbon.r>
	#include "PIGeneral.r"
	#include "PIUtilities.r"
#elif defined(__PIWin__)
	#define Rez
	#include "PIGeneral.h"
	#include "PIUtilities.r"
#endif

#include "ImageLabAutomaticWhiteBalance.hpp"
#include "PIActions.h"

resource 'PiPL' ( 16000, "AutomaticWhiteBalance", purgeable )
{
	{
		Kind { Filter },
		Name { PS_PLUGIN_AWB_NAME "..." },
		Category { PS_PLUGIN_VENDOR_NAME },
		Version { (PS_PLUGIN_MAJOR_VERSION << 16 ) | PS_PLUGIN_AWB_MINOR_VERSION },

		Component { ComponentNumber, PS_PLUGIN_AWB_NAME },

		#ifdef __PIMac__
			CodeMacIntel64 { PS_PLUGIN_ENTRY_POINT_MAC },
		#else
			#if defined(_WIN64)
				CodeWin64X86 { PS_PLUGIN_ENTRY_POINT_WIN64 },
			#else
				CodeWin32X86 { PS_PLUGIN_ENTRY_POINT_WIN32 },
			#endif
		#endif

		SupportedModes
		{
			noBitmap, doesSupportGrayScale,
			noIndexedColor, doesSupportRGBColor,
			doesSupportCMYKColor, doesSupportHSLColor,
			doesSupportHSBColor, doesSupportMultichannel,
			doesSupportDuotone, doesSupportLABColor
		},
		HasTerminology
		{
			PS_PLUGIN_AWB_CLASS_ID,
			PS_PLUGIN_AWB_EVENT_ID,
			16000,
			PS_PLUGIN_AWB_UNIQUE_ID
		},
	
		EnableInfo { "in (PSHOP_ImageMode, RGBMode, GrayScaleMode,"
		             "CMYKMode, HSLMode, HSBMode, MultichannelMode,"
					 "DuotoneMode, LabMode, RGB48Mode, Gray16Mode) ||"
					 "PSHOP_ImageDepth == 16 ||"
					 "PSHOP_ImageDepth == 32" },

		PlugInMaxSize { 2000000, 2000000 },
		
		FilterLayerSupport {doesSupportFilterLayers},
	
		FilterCaseInfo
		{
			{
				/* Flat data, no selection */
				inWhiteMat, outWhiteMat,
				doNotWriteOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
					
				/* Flat data with selection */
				inWhiteMat, outWhiteMat,
				writeOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
				
				/* Floating selection */
				inWhiteMat, outWhiteMat,
				writeOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
					
				/* Editable transparency, no selection */
				inWhiteMat, outWhiteMat,
				doNotWriteOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
					
				/* Editable transparency, with selection */
				inWhiteMat, outWhiteMat,
				writeOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
					
				/* Preserved transparency, no selection */
				inWhiteMat, outWhiteMat,
				doNotWriteOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination,
					
				/* Preserved transparency, with selection */
				inWhiteMat, outWhiteMat,
				writeOutsideSelection,
				filtersLayerMasks, worksWithBlankData,
				copySourceToDestination
			}
		}	
	}
};

