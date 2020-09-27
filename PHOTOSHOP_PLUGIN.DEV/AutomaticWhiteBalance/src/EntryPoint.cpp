#include "ImageLabAutomaticWhiteBalance.hpp"

//-------------------------------------------------------------------------------
// global variables
//-------------------------------------------------------------------------------
// parameters passed into PluginMain that need to be global to the project
FilterRecord* gFilterRecord = nullptr;
intptr_t* gDataHandle = nullptr;
int16* gResult = nullptr;		// all errors go here
SPBasicSuite* sSPBasic = nullptr;


/* ========================================= */
/*  ADOBE PHOTOSHOP PLUGIN ENTRY POINT       */
/* ========================================= */
DLLExport void PluginMain
( 
  const int16 selector,
  FilterRecordPtr filterParamBlock,
  intptr_t* pData,
  int16* pResult
)
{
	FLOATING_POINT_FAST_COMPUTE();

	gFilterRecord = filterParamBlock;
	gDataHandle = pData;
	gResult = pResult;

	if (filterSelectorAbout == selector)
	{
		sSPBasic = (reinterpret_cast<AboutRecord*>(gFilterRecord))->sSPBasic;
	}
	else
	{
		sSPBasic = gFilterRecord->sSPBasic;

		if (nullptr != gFilterRecord->bigDocumentData)
			gFilterRecord->bigDocumentData->PluginUsing32BitCoordinates = true;
	}

	/* do the command according to the selector */
	switch (selector)
	{
		case filterSelectorAbout:
		//	DoAbout();
		break;

		case filterSelectorParameters:
		//	DoParameters();
		break;
	
		case filterSelectorPrepare:
		//	DoPrepare();
		break;

		case filterSelectorStart:
		//	DoStart();
		break;

		case filterSelectorContinue:
		//	DoContinue();
		break;

		case filterSelectorFinish:
		//	DoFinish();
		break;

		default:
		break;
	} /* switch (selector) */

	return;
} /* end PluginMain */