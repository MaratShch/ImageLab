cmake_minimum_required(VERSION 3.5.1)

project (AsyncJob)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set project directories
SET(CMAKE_PROJECT_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SOURCES
	${CMAKE_PROJECT_ROOT_DIRECTORY}/src/*.cpp
)

file(GLOB INCLUDES
	${CMAKE_PROJECT_ROOT_DIRECTORY}/include/*.hpp
	${CMAKE_PROJECT_ROOT_DIRECTORY}/include/*.h
)

add_executable(
	${PROJECT_NAME}
	${SOURCES}
	${INCLUDES}
)

target_include_directories(
	${PROJECT_NAME}
	BEFORE PUBLIC
	${CMAKE_PROJECT_ROOT_DIRECTORY}/include
)

target_compile_options(
	${PROJECT_NAME} BEFORE
	PRIVATE -mtune=intel
	PRIVATE -march=skylake
	PRIVATE -ftree-vectorize
	PRIVATE -fPIC
	PRIVATE -O3
	PRIVATE -m64
)

if(UNIX)
message (STATUS "Build under Linux OS")
find_package (Threads REQUIRED)
target_link_libraries(
	${PROJECT_NAME}
	pthread
	rt
	m
)
else()
message (STATUS "OS isn't Linux like... ${CMAKE_SYSTEM_NAME}")
endif()

