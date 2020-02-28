#include "CThreadPool.hpp"

#ifndef _WINDOWS
#include <unistd.h>
#define Sleep(a)	usleep(a*1000000)
#endif

int main(int argc, char** argv, char** env)
{
	volatile int bRun = 1;

	CThreadPool pool;
	pool.init();

//	CWorkerThread aa;
//
//	aa.signalNewJob();
//	aa.signalNewJob();
//	aa.signalNewJob();
//	aa.signalNewJob();

	while (bRun)
	{
	
		Sleep(100);
	}

	return 0;
}
