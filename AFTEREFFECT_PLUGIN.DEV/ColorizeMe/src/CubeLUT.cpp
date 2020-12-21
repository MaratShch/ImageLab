#include "CubeLUT.h"
#include <iostream>
#include <sstream>


using LUTState = CubeLUT::LUTState;
using tableRow = CubeLUT::tableRow;

CubeLUT::CubeLUT (void)
{ 
	bActive = false;
	uId = 0xDEADBEEF;
	lutName.clear();
	status = NotInitialized; 
}

CubeLUT::~CubeLUT()
{ 
	bActive = false;
	uId = 0xFFFFFFFF;
	lutName.clear();
	status = NotInitialized;
}


std::string CubeLUT::ReadLine (std::ifstream& lutFile, char lineSeperator)
{
	constexpr char CommentMarker = '#';
	std::string textLine{};

	while (0 == textLine.size() || CommentMarker == textLine[0])
	{
		if (lutFile.eof())
		{
			status = PrematureEndOfFile;
			break;
		}

		std::getline (lutFile, textLine, lineSeperator);

		if (lutFile.fail())
		{
			status = ReadError;
			break;
		}
	}

	return textLine;
}


tableRow CubeLUT::ParseTableRow (const std::string& lineOfText)
{
	constexpr int N = 3;

	tableRow f(N);
	std::istringstream line(lineOfText);

	for (int i = 0; i < N; i++)
	{
		line >> f[i];
		if (line.fail())
		{
			status = CouldNotParseTableData;
			break;
		}
	}

	return f;
}


CubeLUT::LUTState CubeLUT::LoadCubeFile(const std::string& fileName)
{
	CubeLUT::LUTState lutState = OK;

	if (!fileName.empty() && fileName != lutName)
	{
		lutName = fileName;
		std::ifstream cubeFile{ lutName };
		if (!cubeFile.good())
		{
			return GenericError;
		}

		lutState = LoadCubeFile(cubeFile);
		cubeFile.close();
	}
	else
		lutState = AlreadyLoaded;

	return lutState;
}


CubeLUT::LUTState CubeLUT::LoadCubeFile (std::ifstream& lutFile)
{
	constexpr char newLineChar = '\n';
	constexpr char carriageReturn = '\r';
	char lineSeparator = newLineChar;

	status = OK;
	title.clear();
	domainMin = tableRow(3, 0.0f);
	domainMax = tableRow(3, 1.0f);

	Lut1D.clear();
	Lut3D.clear();

	for (int i = 0; i < 255; i++)
	{
		const char inc = lutFile.get();
		if (newLineChar == inc)
			break;
		if (carriageReturn == inc)
		{
			if (newLineChar == lutFile.get())
				break;

			lineSeparator = carriageReturn;
			std::cout << "This file uses non-complient line separator \\r (0x0D)" << std::endl;
			break;
		}
		if (i > 250)
		{
			status = LineError;
			break;
		}
	}

	lutFile.seekg(0);
	lutFile.clear();

	int N, CntTitle, CntSize, CntMin, CntMax;
	N = CntTitle = CntSize = CntMin = CntMax = 0;
	
	while (OK == status)
	{
		std::streampos linePos = lutFile.tellg();
		std::string lineOfText = ReadLine (lutFile, lineSeparator);

		if (OK != status)
			break;

		std::istringstream line(lineOfText);
		std::string keyword;
		line >> keyword;

		if ("+" < keyword && keyword < ":")
		{
			lutFile.seekg(linePos);
			break;
		} /* if ("+" < keyword && keyword < ":") */
		else if ("TITLE" == keyword && CntTitle++ == 0)
		{
			constexpr char QUOTE = '"';
			char StartOfTitle;
			line >> StartOfTitle;

			if (QUOTE != StartOfTitle)
			{
				status = TitleMissingQuote;
				break;
			}
			std::getline(line, title, QUOTE);
		} /* else if ("TITLE" == keyword && CntTitle++ == 0) */
		else if ("DOMAIN_MIN" == keyword && CntMin++ == 0)
		{
			line >> domainMin[0] >> domainMin[1] >> domainMin[2];
		} /* else if ("DOMAIN_MIN" == keyword && CntMin++ == 0) */
		else if ("DOMAIN_MAX" == keyword && CntMax++ == 0)
		{
			line >> domainMax[0] >> domainMax[1] >> domainMax[2];
		} /* else if ("DOMAIN_MAX" == keyword && CntMax++ == 0) */
		else if ("LUT_1D_SIZE" == keyword && CntSize++ == 0)
		{
			line >> N;
			if (N < 2 || N > 65536)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			Lut1D = table1D(N, tableRow(3));
		} /* else if ("LUT_1D_SIZE" == keyword && CntSize++ == 0) */
		else if ("LUT_3D_SIZE" == keyword && CntSize++ == 0)
		{
			line >> N;
			if (N < 2 || N > 256)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			Lut3D = table3D(N, table2D(N, table1D(N, tableRow(3))));
		} /* else if ("LUT_3D_SIZE" == keyword && CntSize++ == 0) */
		else
		{
			status = UnknownOrRepeatedKeyword;
			break;
		}

		if (line.fail())
		{
			status = ReadError;
			break;
		}
	} /* while (OK == status) */

	if (OK == status && 0 == CntSize)
	{
		status = LUTSizeOutOfRange;
	}

	if (OK == status && (domainMin[0] > domainMax[0] || domainMin[1] > domainMax[1] || domainMin[2] > domainMax[2]))
	{
		status = DomainBoundReversed;
	}

	/* read lines of LUT data */
	if (Lut1D.size() > 0)
	{
		N = Lut1D.size();
		for (int i = 0; i < N && OK == status; i++)
		{
			Lut1D[i] = ParseTableRow(ReadLine(lutFile, lineSeparator));
		}
	} /* if (Lut1D.size() > 0) */
	else
	{
		N = Lut3D.size();
		for (int b = 0; b < N && OK == status; b++)
		{
			for (int g = 0; g < N && OK == status; g++)
			{
				for (int r = 0; r < N && OK == status; r++)
				{
#ifdef _DEBUG
					std::string lutRow = ReadLine(lutFile, lineSeparator);
					tableRow tblRow = ParseTableRow(lutRow);
					Lut3D[r][g][b] = tblRow;
#else
					Lut3D[r][g][b] = ParseTableRow(ReadLine(lutFile, lineSeparator));
#endif
				} /* for (int r = 0; r < N && OK == status; r++) */
			} /* for (int g = 0; g < N && OK == status; g++) */
		} /* for (int b = 0; b < N && OK == status; b++) */
	}

	bActive = true;
	return status;
}

LUTState CubeLUT::SaveCubeFile(std::ofstream& lutFile)
{
	/* not mplemented, because not required yet */
	return GenericError;
}

//tableRow CubeLUT::interpNearest(const rgbVec& in)
//{
//	return Lut3D [NEAR(in.r)][NEAR(in.g)][NEAR(in.b)];
//}

