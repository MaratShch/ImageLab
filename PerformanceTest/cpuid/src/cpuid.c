#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32 || _WIN64

//  Windows
#define cpuid(info, x)    __cpuidex(info, x, 0)

#else

//  GCC Intrinsics
#include <cpuid.h>
void cpuid(int info[4], int InfoType){
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

#endif

#ifndef bool
 #define bool unsigned char
#endif

void main (void)
{
unsigned cores = 0;
unsigned cpuFeatures = 0;
unsigned logical = 0;

char vendor[16] = { 0 };

//  Misc.
bool HW_MMX;
bool HW_x64;
bool HW_ABM;      // Advanced Bit Manipulation
bool HW_RDRAND;
bool HW_BMI1;
bool HW_BMI2;
bool HW_ADX;
bool HW_PREFETCHWT1;

//  SIMD: 128-bit
bool HW_SSE;
bool HW_SSE2;
bool HW_SSE3;
bool HW_SSSE3;
bool HW_SSE41;
bool HW_SSE42;
bool HW_SSE4a;
bool HW_AES;
bool HW_SHA;

//  SIMD: 256-bit
bool HW_AVX;
bool HW_XOP;
bool HW_FMA3;
bool HW_FMA4;
bool HW_AVX2;

//  SIMD: 512-bit
bool HW_AVX512F;    //  AVX512 Foundation
bool HW_AVX512CD;   //  AVX512 Conflict Detection
bool HW_AVX512PF;   //  AVX512 Prefetch
bool HW_AVX512ER;   //  AVX512 Exponential + Reciprocal
bool HW_AVX512VL;   //  AVX512 Vector Length Extensions
bool HW_AVX512BW;   //  AVX512 Byte + Word
bool HW_AVX512DQ;   //  AVX512 Doubleword + Quadword
bool HW_AVX512IFMA; //  AVX512 Integer 52-bit Fused Multiply-Add
bool HW_AVX512VBMI; //  AVX512 Vector Byte Manipulation Instructions

bool hyperThreads;

int info[4] = { 0 };
cpuid(info, 0);
int nIds = info[0];

((unsigned *)vendor)[0] = info[1]; // EBX
((unsigned *)vendor)[1] = info[3]; // EDX
((unsigned *)vendor)[2] = info[2]; // ECX
vendor[12] = '\0';

// Get CPU features
cpuid(info, 1);
cpuFeatures = info[3]; // EDX
								// Logical core count per CPU
cpuid(info, 1);
logical = (info[1] >> 16) & 0xff; // EBX[23:16]
printf(" logical cpus: %d\n", logical);
cores = logical;

if (0 == strcmp(vendor, "GenuineIntel")) {
	// Get DCP cache info
	cpuid(info, 11);
	cores = ((info[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
}
else if (0 == strcmp(vendor, "AuthenticAMD")) {
	// Get NC: Number of CPU cores - 1
	cpuid(info, 0x80000008);
	cores = ((unsigned)(info[2] & 0xff)) + 1; // ECX[7:0] + 1
}

hyperThreads = cpuFeatures & (1 << 28) && cores < logical;

cpuid(info, 0x80000000);
unsigned nExIds = info[0];

//  Detect Features
if (nIds >= 0x00000001){
    cpuid(info,0x00000001);
    HW_MMX    = (info[3] & ((int)1 << 23)) != 0;
    HW_SSE    = (info[3] & ((int)1 << 25)) != 0;
    HW_SSE2   = (info[3] & ((int)1 << 26)) != 0;
    HW_SSE3   = (info[2] & ((int)1 <<  0)) != 0;

    HW_SSSE3  = (info[2] & ((int)1 <<  9)) != 0;
    HW_SSE41  = (info[2] & ((int)1 << 19)) != 0;
    HW_SSE42  = (info[2] & ((int)1 << 20)) != 0;
    HW_AES    = (info[2] & ((int)1 << 25)) != 0;

    HW_AVX    = (info[2] & ((int)1 << 28)) != 0;
    HW_FMA3   = (info[2] & ((int)1 << 12)) != 0;

    HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;

}
if (nIds >= 0x00000007){
    cpuid(info,0x00000007);
    HW_AVX2   = (info[1] & ((int)1 <<  5)) != 0;

    HW_BMI1        = (info[1] & ((int)1 <<  3)) != 0;
    HW_BMI2        = (info[1] & ((int)1 <<  8)) != 0;
    HW_ADX         = (info[1] & ((int)1 << 19)) != 0;
    HW_SHA         = (info[1] & ((int)1 << 29)) != 0;
    HW_PREFETCHWT1 = (info[2] & ((int)1 <<  0)) != 0;

    HW_AVX512F     = (info[1] & ((int)1 << 16)) != 0;
    HW_AVX512CD    = (info[1] & ((int)1 << 28)) != 0;
    HW_AVX512PF    = (info[1] & ((int)1 << 26)) != 0;
    HW_AVX512ER    = (info[1] & ((int)1 << 27)) != 0;
    HW_AVX512VL    = (info[1] & ((int)1 << 31)) != 0;
    HW_AVX512BW    = (info[1] & ((int)1 << 30)) != 0;
    HW_AVX512DQ    = (info[1] & ((int)1 << 17)) != 0;
    HW_AVX512IFMA  = (info[1] & ((int)1 << 21)) != 0;
    HW_AVX512VBMI  = (info[2] & ((int)1 <<  1)) != 0;
}
if (nExIds >= 0x80000001){
    cpuid(info,0x80000001);
    HW_x64   = (info[3] & ((int)1 << 29)) != 0;
    HW_ABM   = (info[2] & ((int)1 <<  5)) != 0;
    HW_SSE4a = (info[2] & ((int)1 <<  6)) != 0;
    HW_FMA4  = (info[2] & ((int)1 << 16)) != 0;
    HW_XOP   = (info[2] & ((int)1 << 11)) != 0;
}



printf("Vendor:        %s\n", vendor);
printf("Physical cores = %d\n", cores);
printf("HW_MMX         = %d\n", HW_MMX);
printf("HW_SSE         = %d\n", HW_SSE);
printf("HW_SSE2        = %d\n", HW_SSE2);
printf("HW_SSE3        = %d\n", HW_SSE3);
printf("HW_SSSE3       = %d\n", HW_SSSE3);
printf("HW_SSE41       = %d\n", HW_SSE41);
printf("HW_SSE42       = %d\n", HW_SSE42);
printf("HW_AES         = %d\n", HW_AES);
printf("HW_AVX         = %d\n", HW_AVX);
printf("HW_FMA3        = %d\n", HW_FMA3);
printf("HW_RDRAND      = %d\n", HW_RDRAND);
printf("HW_AVX2        = %d\n", HW_AVX2);
printf("HW_BMI1        = %d\n", HW_BMI1);
printf("HW_BMI2        = %d\n", HW_MMX);
printf("HW_ADX         = %d\n", HW_ADX);
printf("HW_SHA         = %d\n", HW_SHA);
printf("HW_PREFETCHWT1 = %d\n", HW_PREFETCHWT1);
printf("HW_AVX512F     = %d\n", HW_AVX512F);
printf("HW_AVX512CD    = %d\n", HW_AVX512CD);
printf("HW_AVX512PF    = %d\n", HW_AVX512PF);
printf("HW_AVX512ER    = %d\n", HW_AVX512ER);
printf("HW_AVX512VL    = %d\n", HW_AVX512VL);
printf("HW_AVX512BW    = %d\n", HW_AVX512BW);
printf("HW_AVX512DQ    = %d\n", HW_AVX512DQ);
printf("HW_AVX512IFMA  = %d\n", HW_AVX512IFMA);
printf("HW_AVX512VBMI  = %d\n", HW_AVX512VBMI);
printf("HW_x64         = %d\n", HW_x64);
printf("HW_ABM         = %d\n", HW_ABM);
printf("HW_SSE4a       = %d\n", HW_SSE4a);
printf("HW_FMA4        = %d\n", HW_FMA4);
printf("HW_XOP         = %d\n", HW_XOP);

printf("hyper - threads: %s\n", hyperThreads ? "true" : "false");

printf("Complete. Press <ENTER> for exit...\n");
(void)getchar();

return;
}

