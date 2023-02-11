#include "ImageLabUtils.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\Debug\\ImageLabUtils.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\Release\\ImageLabUtils.lib")
#endif

using namespace ImageLabMemoryUtils;


int main(void)
{
	MemoryInterface* pInstance = MemoryInterface::getInstance();

	return 0;
}