REM https://stackoverflow.com/questions/14932315/how-to-compile-qt-5-under-windows-or-linux-32-or-64-bit-static-or-dynamic-on-v
REM QT sources: http://download.qt.io/official_releases/

set QT_VERSION=5.15.9
rm -fr %QT_VERSION%_build
 
cls

set QTDIR= 
set QMAKESPEC=
set QMAKEPATH=
set XQMAKESPEC=

set QTDIR=C:\QT\%QT_VERSION%\qtbase
set FBXSDK=C:\Autodesk\FBX_SDK\2020.2
set FBXSDK_LIBS=%FBXSDK%\lib\vs2015\x64\release\libfbxsdk.lib
set VULKAN_SDK=C:\VulkanSDK\1.2.170.0
set PATH=%QTDIR%\bin;%FBXSDK%;%PATH%

mkdir %QT_VERSION%_build
cd %QT_VERSION%_build

..\%QT_VERSION%\configure -verbose -recheck-all -mp -debug-and-release  -force-debug-info -separate-debug-info -opensource -confirm-license -platform win32-msvc -avx2 -qt3d-simd avx2 -feature-vulkan -qt-zlib -qt-libpng -qt-libjpeg -opengl es2 -angle -I "%VULKAN_SDK%\Include" -I "%FBXSDK%\include" -L "%VULKAN_SDK%\Lib" -L "%FBXSDK_LIBS%" -D "%VULKAN_SDK%\Bin" -nomake examples -nomake tests

..\jom\jom -j 8