REM !!!!! NOT WORKING !!!!
REM https://stackoverflow.com/questions/14932315/how-to-compile-qt-5-under-windows-or-linux-32-or-64-bit-static-or-dynamic-on-v
REM QT sources: http://download.qt.io/official_releases/
REM run "x64 Native Tools Command Prompt" for build QT from Windows/Startup->VisualStudio 2015
 
cls

rm -fr qt6_build

set QTDIR= 
set QMAKESPEC=
set QMAKEPATH=
set XQMAKESPEC=

set QTDIR=C:\QT\6.5.0\qtbase
set FBXSDK=C:\Autodesk\FBX_SDK\2020.3.2
set FBXSDK_LIBS=%FBXSDK%\lib\vs2015\x64\release\libfbxsdk.lib
set VULKAN_SDK=C:\C:\VulkanSDK\1.3.243.0
set PATH=%QTDIR%\bin;%FBXSDK%;%PATH%

mkdir qt6_build
cd qt6_build

..\6.5.0\configure  -verbose -release -force-debug-info -separate-debug-info -opensource -confirm-license -platform win32-msvc -avx2 -qt3d-simd avx2 -feature-vulkan -qt-zlib -qt-libpng -qt-libjpeg -qt-freetype -opengl desktop -I "%VULKAN_SDK%\Include" -I "%FBXSDK%\include" -L "%VULKAN_SDK%\Lib" -L "%FBXSDK_LIBS%" -D "%VULKAN_SDK%\Bin" -nomake examples -nomake tests

..\jom\jom -j 8