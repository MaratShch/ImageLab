#pragma once

#ifdef _WINDLL
 #ifdef _IMAGELAB_DLL_LINK 
  #define DLL_LINK __declspec(dllimport)
 #else
  #define DLL_LINK __declspec(dllexport)
#endif
#else
 #define DLL_LINK __declspec(dllimport)
#endif

#ifndef DLL_API_EXPORT
 #define DLL_API_EXPORT  extern "C" DLL_LINK
#endif