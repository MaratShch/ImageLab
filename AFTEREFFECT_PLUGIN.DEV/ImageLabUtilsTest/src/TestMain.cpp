#include "ImageLabUtils.hpp"
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\Debug\\ImageLabUtils.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\Release\\ImageLabUtils.lib")
#endif

using namespace ImageLabMemoryUtils;


int main(void)
{
	CMemoryInterface* pInstance1 = CMemoryInterface::getInstance();
	CMemoryInterface* pInstance2 = CMemoryInterface::getInstance();

	std::cout << "First instance: " << pInstance1 << " Second instance: " << pInstance2 << std::endl;
	
	const int32_t memSize1 = 1920 * 1080 * 3;
	const int32_t memSize2 = 3840 * 2060 * 3;

	void* ptr1 = nullptr;
	void* ptr2 = nullptr;
	int32_t id1 = -1;
	int32_t id2 = -1;

	id2 = pInstance1->allocMemoryBlock(memSize2, &ptr2);
	pInstance1->releaseMemoryBlock(id2);

	id1 = pInstance1->allocMemoryBlock(memSize1, &ptr1);
	id2 = pInstance2->allocMemoryBlock(memSize1, &ptr2);

	return 0;
}