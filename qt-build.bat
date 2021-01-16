REM https://stackoverflow.com/questions/14932315/how-to-compile-qt-5-under-windows-or-linux-32-or-64-bit-static-or-dynamic-on-v

cls

set QTDIR= 
set QMAKESPEC=
set QMAKEPATH=
set XQMAKESPEC=

set QTDIR=C:\QT\5.15.2\qtbase
set PATH=%QTDIR%\bin;%PATH%

mkdir qt5_build
cd qt5_build

..\5.15.2\configure -debug-and-release -opensource -platform win32-msvc -qt-zlib -qt-libpng -qt-libjpeg -opengl desktop

..\jom\jom -j 8
