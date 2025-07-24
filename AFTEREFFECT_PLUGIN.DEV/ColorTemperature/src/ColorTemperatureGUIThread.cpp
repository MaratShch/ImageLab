#include <atomic>
#include <windows.h>

std::atomic<bool> guiThreadAlive{ true };
static HANDLE h{};
static HANDLE hExit{};

#ifdef _DEBUG
std::size_t drawEvents = 0u;
#endif

DWORD WINAPI guiDrawHandle (LPVOID lpParam)
{
    int32_t forceDrawEvent = 10;

    while (true == guiThreadAlive)
    {
        Sleep (50u);
        if (forceDrawEvent <= 0 && true == guiThreadAlive)
        {
            /* restore delay counter */
            forceDrawEvent = 10;

#ifdef _DEBUG
            drawEvents++;
#endif

            /* force redraw CCT */

        }
        forceDrawEvent--;
    }

    SetEvent (hExit);
    return EXIT_SUCCESS;
}


void StartGuiThread(void)
{
    hExit = CreateEvent(NULL, FALSE, FALSE, NULL);
    h = CreateThread(NULL, 0ull, reinterpret_cast<LPTHREAD_START_ROUTINE>(guiDrawHandle), NULL, 0, NULL);
    return;
}

void StopGuiThread(void)
{
    guiThreadAlive = false;
    WaitForSingleObject(hExit, 500u);
    CloseHandle (hExit);
    CloseHandle (h);
    return;
}