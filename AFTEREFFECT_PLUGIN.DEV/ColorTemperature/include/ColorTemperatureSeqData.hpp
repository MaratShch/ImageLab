#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_SEQUENCE_DATA__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_SEQUENCE_DATA__

#include "ColorTemperatureProc.hpp"

constexpr uint32_t sequenceDataMagic = 0xDEADBEEF;

#pragma pack(push)
#pragma pack(1)

typedef struct {
	bool isFlat;
	uint32_t magic;
	rgbCoefficients colorCoeff;
}flatSequenceData;

using unflatSequenceData = flatSequenceData;

#pragma pack(pop)

constexpr size_t flatSequenceDataSize   = sizeof(flatSequenceData);
constexpr size_t unflatSequenceDataSize = sizeof(unflatSequenceData);


#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_SEQUENCE_DATA__ */