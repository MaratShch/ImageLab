#include "ImageLabAutomaticWhteBalance.hpp"

DLLExport void PluginMain
( 
  const int16 selector,
  FilterRecordPtr filterParamBlock,
  intptr_t* pData,
  int16* pResult
)
{
	FLOATING_POINT_FAST_COMPUTE();

	return;
} // end PluginMain