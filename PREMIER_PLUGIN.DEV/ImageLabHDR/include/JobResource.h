#pragma once

#ifndef CPU_MEM_ALIGNMENT
 #define CPU_MEM_ALIGNMENT	0x1000
#endif

class JobResource 
{
private:

	unsigned int*   histogramBufferPtr;
	unsigned char*  binBufferPtr;
	unsigned short* cumSumBufferPtr;

	unsigned int*   histogramBufferPtrOrig;
	unsigned char*  binBufferPtrOrig;
	unsigned short* cumSumBufferPtrOrig;

	DWORD hitogramBufferSize;
	DWORD binBufferSize;
	DWORD cumSumBufferSize;

	unsigned int    coreNumber;

	bool resourceValid;

public:
	JobResource();
	JobResource(const unsigned int& coreNum);

	virtual ~JobResource();

	virtual bool isResourceValid(void) {
		return resourceValid;
	}

	unsigned int* getHistogramBuffer(void) {
		return histogramBufferPtr;
	}

	unsigned char* getBinBuffer(void) {
		return binBufferPtr;
	}

	unsigned short* getCumSumBuffer(void) {
		return cumSumBufferPtr;
	}

	bool allocateHistogramBuffer(unsigned int alignment);
	void freeHistogramBuffer(void);

	bool allocateBinBuffer(unsigned int alignment);
	void freeBinBuffer(void);

	bool allocateCumSumBuffer(unsigned int alignment);
	void freeeCumSumBuffer(void);

	bool allocateMemoryResources(void);
	void freeMemoryResources(void);

protected:

};

