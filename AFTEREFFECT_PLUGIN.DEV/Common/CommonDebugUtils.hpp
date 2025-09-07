#pragma once

//#include <windows.h>
#include <iostream>
#include <sstream>
#include <fstream>


#ifdef _DEBUG

#if 0
template <typename T, typename U>
inline void DBOUT (const T& val, const U& _dbg_function, const std::string& _dbg_file = __FILE__, const int& _dbg_line = __LINE__) 
{                             
  std::ostringstream os_;     
  os_ << _dbg_file << "." << _dbg_line << " [" << _dbg_function << "] " << val << std::endl;
  OutputDebugString( os_.str().c_str() );
  return;
}
#endif


template <typename T>
inline bool dbgFileSave
(
    const char* strFileName,   // zero terminated file name 
    const T* pData,            // data to save
    uint32_t sizeX,            // horizontal size in elements
    uint32_t sizeY,            // vertical size in elements
    uint32_t lPitch            // line pitch in elements
)
{
    A_long j, i;
    if (nullptr != strFileName && nullptr != pData)
    {
        const size_t lineBytesSize = sizeX * sizeof(T);
        std::ofstream file (strFileName, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open file");
         for (j = 0; j < sizeY; j++)
         {
            file.write(reinterpret_cast<const char*>(pData + j * lPitch), lineBytesSize);
            if (!file.good())
               return false;
         }
    }
    return true;
}

template <typename T>
inline bool dbgFileSave
(
    const char* strFileName,   // zero terminated file name 
    const T* pData,            // data to save
    uint32_t sizeX,            // horizontal size in elements
    uint32_t sizeY             // vertical size in elements
)
{
    return dbgFileSave(strFileName, pData, sizeX, sizeY, sizeX);
}

template <typename T>
inline bool dbgFileSave
(
     const char* strFileName,   // zero terminated file name 
     const T* pData,            // data to save
     uint32_t size              // size in elements
)
{
    return dbgFileSave(strFileName, pData, size, 1, size);
}


#else

  #define DBOUT( s , t )

#endif // _DEBUG
