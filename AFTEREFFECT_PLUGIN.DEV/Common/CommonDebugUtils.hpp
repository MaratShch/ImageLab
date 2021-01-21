#pragma once

#include <windows.h>
#include <iostream>
#include <sstream>

#ifdef _DEBUG
 template <typename T, typename U>
 void DBOUT (const T& val, const U& _dbg_function, const std::string& _dbg_file = __FILE__, const int& _dbg_line = __LINE__) 
 {                             
   std::ostringstream os_;     
   os_ << _dbg_file << "." << _dbg_line << " [" << _dbg_function << "] " << val << std::endl;
   OutputDebugString( os_.str().c_str() );
   return;
 }
#else
  #define DBOUT( s , t )
#endif
