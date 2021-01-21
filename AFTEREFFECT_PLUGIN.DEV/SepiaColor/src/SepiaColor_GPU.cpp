#include "SepiaColor.hpp"
#include "ColorTransformMatrix.hpp"

#include "ImageLab2GpuObj.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif

#if 0
DllExport
prSuiteError xGPUFilterEntry(
	csSDK_uint32 inHostInterfaceVersion,
	csSDK_int32* ioIndex,
	prBool inStartup,
	piSuitesPtr piSuites,
	PrGPUFilter* outFilter,
	PrGPUFilterInfo* outFilterInfo) 
{
	PF_Err		err = PF_Err_NONE;

}
#endif