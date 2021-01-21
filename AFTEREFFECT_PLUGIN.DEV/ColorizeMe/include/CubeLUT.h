#ifndef __CUBE_LUT__
#define __CUBE_LUT__

#include <string>
#include <vector>
#include <fstream>
#include <atomic>
#include "ClassRestrictions.hpp"


struct rgbVec
{
	float r;
	float g;
	float b;
};

/* This is classic CUBE LUT file load implementation proposed by ADOBE company.
   Let's check how we can optimize it for performance
*/
class CubeLUT final
{
//	CLASS_NON_MOVABLE (CubeLUT);
//	CLASS_NON_COPYABLE(CubeLUT);

public:

	typedef std::string lutFileName;
	typedef std::vector<float> tableRow;
	typedef std::vector<tableRow> table1D;
	typedef std::vector<table1D>  table2D;
	typedef std::vector<table2D>  table3D;

	enum LUTState
	{
		AlreadyLoaded = -1,
		OK = 0,
		NotInitialized,
		ReadError = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		UnknownOrRepeatedKeyword = 20,
		TitleMissingQuote,
		DomainBoundReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData,
		GenericError = 100
	};
	
	unsigned int uId;
	lutFileName lutName;
	LUTState status;
	std::string title;
	tableRow domainMin;
	tableRow domainMax;
	table1D  Lut1D;
	table3D  Lut3D;

	CubeLUT(void);
	~CubeLUT();

	const std::string& GetLutName(void) { return lutName; }

	LUTState LoadCubeFile (const std::string& fileName);
	LUTState LoadCubeFile (std::ifstream& lutFile);
	LUTState SaveCubeFile (std::ofstream& lutFile);

	bool LutIsLoaded(void) { return bActive; }

	int32_t GetLutSize(void) { return lutSize; }
//	tableRow interpNearest(const rgbVec& in);

	uint32_t increaseRefCnt(void) { return ++refCnt; }
	uint32_t decreseRefCnt(void) { return (refCnt > 0u) ? --refCnt : 0u; }

private:
	int32_t lutSize;
	bool bActive = false;
	std::atomic<uint32_t> refCnt;
	std::string ReadLine (std::ifstream& lutFile, char lineSeperator);
	tableRow ParseTableRow(const std::string& lineOfText);
};

constexpr size_t LUT_OBJ_SIZE = sizeof(CubeLUT);

using LutObjHndl = CubeLUT*;

#endif