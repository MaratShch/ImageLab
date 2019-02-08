#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
