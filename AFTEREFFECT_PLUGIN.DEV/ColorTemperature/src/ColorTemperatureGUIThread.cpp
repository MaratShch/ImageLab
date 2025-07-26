#include <atomic>
#include <windows.h>
#include "ColorTemperatureDraw.hpp"

static std::atomic<bool> needsRedraw{ true };
static std::atomic<bool> guiThreadAlive{ true };
static std::atomic<AlgoProcT> gui_CCT{ 6500.f };
static std::atomic<AlgoProcT> gui_Duv{ 0.f };

static HANDLE h{};
static HANDLE hExit{};

#ifdef _DEBUG
std::size_t drawEvents = 0u;
#endif


void SetGUI_CCT (const std::pair<AlgoProcT, AlgoProcT>& cct_duv) noexcept
{
    gui_CCT.exchange (cct_duv.first);
    gui_Duv.exchange (cct_duv.second);
    return;
}

void ForceRedraw (void) noexcept
{
    needsRedraw.store(true);
    return;
}

bool isRedraw (void) noexcept
{
    const bool redrawFlag = needsRedraw.load();
    return redrawFlag;
}

bool ProcRedrawComplete (void) noexcept
{
    const bool redrawFlag = needsRedraw.exchange(false);
    return redrawFlag;
}

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
            ForceRedraw();
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
