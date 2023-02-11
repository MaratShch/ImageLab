#pragma once

#ifdef _WINDLL
 #define DLL_LINK __declspec(dllexport)
#else
 #define DLL_LINK __declspec(dllimport)
#endif

#ifndef DLL_EXPORT
 #define DLL_EXPORT __declspec(dllexport)
#endif

#ifndef DLL_API_EXPORT
 #define DLL_API_EXPORT  extern "C" DLL_EXPORT
#endif