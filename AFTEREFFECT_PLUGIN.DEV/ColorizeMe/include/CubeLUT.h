#ifndef __CUBE_LUT__
#define __CUBE_LUT__

#include <string>
#include <vector>
#include <fstream>
#include "ClassRestrictions.hpp"

#define NEAR(x)	((int)((x)+0.5))

struct rgbVec
{
	float r;
	float g;
	float b;
};

/* This is classic CUBE LUT file load implementation proposed by ADOBE company.
   Let's check how we can optimize it for performance
*/
class CubeLUT
{
	CLASS_NON_MOVABLE (CubeLUT);
	CLASS_NON_COPYABLE(CubeLUT);

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
	
	lutFileName lutName;
	LUTState status;
	std::string title;
	tableRow domainMin;
	tableRow domainMax;
	table1D  Lut1D;
	table3D  Lut3D;

	CubeLUT(void) { lutName.clear(); status = NotInitialized; }
	virtual ~CubeLUT() { lutName.clear(); status = NotInitialized;	}

	LUTState LoadCubeFile(const std::string& fileName);
	LUTState LoadCubeFile (std::ifstream& lutFile);
	LUTState SaveCubeFile (std::ofstream& lutFile);

	bool LutIsLoaded(void) { return bActive; }

//	tableRow interpNearest(const rgbVec& in);


private:
	bool bActive = false;
	std::string ReadLine (std::ifstream& lutFile, char lineSeperator);
	tableRow ParseTableRow(const std::string& lineOfText);
};


#endif