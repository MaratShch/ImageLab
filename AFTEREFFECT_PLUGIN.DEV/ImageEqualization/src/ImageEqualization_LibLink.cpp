#include <windows.h>

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\Debug\\Avx2ProcLib.lib")     // static lib
#else 
#pragma comment(lib, "..\\BUILD.OUT\\Release\\Avx2ProcLib.lib")   // static lib
#endif
