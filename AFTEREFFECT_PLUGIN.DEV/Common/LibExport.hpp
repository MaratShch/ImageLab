#pragma once

#ifndef DLL_EXPORT
 #define DLL_EXPORT __declspec(dllexport)
#endif

#ifndef DLL_API_EXPORT
 #define DLL_API_EXPORT  extern "C" DLL_EXPORT
#endif