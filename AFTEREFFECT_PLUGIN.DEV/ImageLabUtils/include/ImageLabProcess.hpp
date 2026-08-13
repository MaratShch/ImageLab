#pragma once

#include <cstdint>
#include <string>
#include <windows.h>
#include "Common.hpp"
#include "ImageLabCommonUtils.hpp"


constexpr std::uint64_t magic1 = Magic('I', 'M', 'A', 'G', 'E', 'L', 'a', 'b');

template <std::size_t SIZE>
using CmdArgs = std::array<std::wstring, SIZE>;


template <std::size_t SIZE_CMD_LINE_ARGS>
inline PROCESS_INFORMATION ImageLabRunExecutable (const std::wstring& exe_full_path, const CmdArgs<SIZE_CMD_LINE_ARGS>& cmdLine)
{
    CACHE_ALIGN STARTUPINFOW si{ 0 };
    PROCESS_INFORMATION pi{ 0 };

    si.cb = sizeof(si); // Must set the size!

    std::wstring magicStr = std::to_wstring(magic1);

    // 1. Flatten the array into a single command line string.
    // We start with the executable path (wrapped in quotes) so argv[0] is correct.
    std::wstring flatCmdLine = L"\"" + exe_full_path + L"\"";
    flatCmdLine += L" ";
    flatCmdLine += magicStr;

    // 2. Append all arguments, separated by spaces.
    for (std::size_t i = 0; i < SIZE_CMD_LINE_ARGS; ++i)
    {
        flatCmdLine += L" ";
        // Note: If your arguments might contain spaces (like file paths), 
        // you should wrap them in quotes here: L"\"" + args[i] + L"\""
        flatCmdLine += args[i];
    }
    // Launch the process
    BOOL success = CreateProcessW
    (
        exe_full_path.c_str(),  // 1. lpApplicationName (Can safely be a const wchar_t*)
        &flatCmdLine[0],        // 2. lpCommandLine (MUST STILL BE MUTABLE!)
        NULL,                   // 3. Process attributes
        NULL,                   // 4. Thread attributes
        FALSE,                  // 5. Inherit handles
        0,                      // 6. Creation flags
        NULL,                   // 7. Environment
        NULL,                   // 8. Current directory
        &si,                    // 9. Startup info
        &pi                     // 10. Process information
    );

    // Optional: You might want to log GetLastError() if success == FALSE
    if (FALSE == success)
        pi = {};

    return pi;
}


void ImageLabStopExecutable (PROCESS_INFORMATION& pi);