#include <windows.h>

#ifdef _DEBUG
#pragma comment(lib, "../Debug/AsyncJob.lib")
#else
#pragma comment(lib, "../Release/AsyncJob.lib")
#endif