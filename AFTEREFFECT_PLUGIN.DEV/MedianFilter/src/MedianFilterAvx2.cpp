#include "MedianFilter.hpp"

/* static link with AVX2 library */
#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\Avx2ProcLib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\Avx2ProcLib.lib")
#endif

