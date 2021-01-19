REM https://stackoverflow.com/questions/14932315/how-to-compile-qt-5-under-windows-or-linux-32-or-64-bit-static-or-dynamic-on-v
REM QT sources: http://download.qt.io/official_releases/
 
cls

set QTDIR= 
set QMAKESPEC=
set QMAKEPATH=
set XQMAKESPEC=

set QTDIR=C:\QT\5.15.2\qtbase
set FBXSDK=C:\Autodesk\FBX_SDK\2020.2
set FBXSDK_LIBS=%FBXSDK%\lib\vs2015\x64\release\libfbxsdk.lib

set PATH=%QTDIR%\bin;%FBXSDK%;%PATH%

mkdir qt5_build
cd qt5_build

..\5.15.2\configure -verbose -debug-and-release -opensource -confirm-license -platform win32-msvc -avx2 -qt3d-simd avx2 -feature-vulkan -qt-zlib -qt-libpng -qt-libjpeg -opengl dynamic -I "C:\VulkanSDK\1.2.162.1\Include" -I "%FBXSDK%\include" -L "C:\VulkanSDK\1.2.162.1\Lib" -L "%FBXSDK_LIBS%" -D "C:\VulkanSDK\1.2.162.1\Bin" -nomake examples -nomake tests

..\jom\jom -j 8