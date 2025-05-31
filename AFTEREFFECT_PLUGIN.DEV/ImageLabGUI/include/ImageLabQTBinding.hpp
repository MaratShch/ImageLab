#ifndef __IMAGE_LAB_APPLICATION_QT_APPLICATION_BINDING__
#define __IMAGE_LAB_APPLICATION_QT_APPLICATION_BINDING__

#include "LibExport.hpp"
#include <QApplication>

DLL_API_EXPORT QApplication* AllocQTApplication (void);
DLL_API_EXPORT void FreeQTApplication (void);

#endif // __IMAGE_LAB_APPLICATION_QT_APPLICATION_BINDING__