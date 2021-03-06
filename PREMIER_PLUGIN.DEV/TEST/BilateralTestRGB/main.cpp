//#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "ImageLabBilateral.h"

#if 0
typedef unsigned long       DWORD;
typedef long				LONG;
typedef unsigned short      WORD;

#undef FAR
#undef  NEAR
#define FAR
#define NEAR

#pragma pack(push, 1)

typedef struct tagBITMAPINFOHEADER {
	DWORD      biSize;
	LONG       biWidth;
	LONG       biHeight;
	WORD       biPlanes;
	WORD       biBitCount;
	DWORD      biCompression;
	DWORD      biSizeImage;
	LONG       biXPelsPerMeter;
	LONG       biYPelsPerMeter;
	DWORD      biClrUsed;
	DWORD      biClrImportant;
} BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;

typedef struct tagBITMAPFILEHEADER {
	WORD    bfType;
	DWORD   bfSize;
	WORD    bfReserved1;
	WORD    bfReserved2;
	DWORD   bfOffBits;
} BITMAPFILEHEADER, FAR *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;

#pragma pack(pop)
#endif

unsigned char* LoadBitmapFile(const char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr; //our file pointer
	BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
	unsigned char *bitmapImage;  //store image data
	DWORD imageIdx = 0;  //image index counter
	unsigned char tempRGB;  //our swap variable

							//open filename in read binary mode
	fopen_s(&filePtr, filename, "rb");
	if (filePtr == NULL)
		return NULL;

	//read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	//verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.bfType != 0x4D42)
	{
		fclose(filePtr);
		return NULL;
	}

	//read the bitmap info header
	fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr); // small edit. forgot to add the closing bracket at sizeof

																   //move file point to the begging of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	//allocate enough memory for the bitmap image data
	bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);

	//verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return NULL;
	}

	//read in the bitmap image data
	fread(bitmapImage, bitmapInfoHeader->biSizeImage, 1, filePtr);

	//make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
	}

	//swap the r and b values to get RGB (bitmap is BGR)
	for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage; imageIdx += 3) // fixed semicolon
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	//close file and return bitmap iamge data
	fclose(filePtr);
	return bitmapImage;
}


int main(void)
{
	volatile bool bMainProcSim = true;
	int idx = 10;

	//const char* path = { "..\\IMG.IN\\ImgRef_1024x768xRGB.bmp" };
	const char* path = { "..\\IMG.IN\\ImgRef_128x96xRGB.bmp" };
	BITMAPINFOHEADER header = { 0 };
	unsigned char* pBmp = nullptr;

	// simulate DLL' attachment
	DllMain(nullptr, DLL_PROCESS_ATTACH, nullptr);

	pBmp = LoadBitmapFile(path, &header);

	const int sizeX = header.biWidth;
	const int sizeY = header.biHeight;

	printf("Complete...\n");
	return 0;
}