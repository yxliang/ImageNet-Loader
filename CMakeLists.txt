
cmake_minimum_required (VERSION 3.8)
project (ImageNet-Loader)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_DEBUG_POSTFIX "d")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
# set(CMAKE_CXX_FLAGS "-std=c++17") # -Wextra
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O2")

## external lib paths
SET(3rdlib_DIR "D:/Libraries")
#gflags
add_library(gflags STATIC IMPORTED GLOBAL)
set_target_properties(gflags PROPERTIES IMPORTED_LOCATION_DEBUG "${3rdlib_DIR}/gflags/lib/gflags_static_debug.lib" 
										IMPORTED_LOCATION_RELEASE "${3rdlib_DIR}/gflags/lib/gflags_static.lib" 
										INTERFACE_COMPILE_DEFINITIONS "GFLAGS_IS_A_DLL=0"
										INTERFACE_INCLUDE_DIRECTORIES "${3rdlib_DIR}/gflags/include"
										INTERFACE_LINK_LIBRARIES "shlwapi.lib")
#glog
add_library(glog STATIC IMPORTED GLOBAL)
set_target_properties(glog PROPERTIES IMPORTED_LOCATION_DEBUG "${3rdlib_DIR}/glog/lib/glogd.lib"
									  IMPORTED_LOCATION_RELEASE "${3rdlib_DIR}/glog/lib/glog.lib"
									  INTERFACE_COMPILE_DEFINITIONS "GFLAGS_DLL_DECLARE_FLAG=;GFLAGS_DLL_DEFINE_FLAG=;GLOG_NO_ABBREVIATED_SEVERITIES;GOOGLE_GLOG_DLL_DECL="
									  INTERFACE_INCLUDE_DIRECTORIES "${3rdlib_DIR}/glog/include")


#opencv										
SET(OpenCV_DIR "${3rdlib_DIR}/OpenCV")
option(OpenCV_STATIC "Use opencv static"   OFF)										
FIND_PACKAGE( OpenCV REQUIRED COMPONENTS opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui opencv_features2d opencv_objdetect opencv_stitching opencv_video opencv_videoio opencv_videostab opencv_line_descriptor opencv_xfeatures2d opencv_ximgproc opencv_xobjdetect opencv_xphoto )		
IF (${OpenCV_VERSION} VERSION_LESS 3.0)
	MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}")
ENDIF()	

list(APPEND LIGHTSTEP_LINK_LIBRARY ${OpenCV_LIBS})

## other link libs
list(APPEND LIGHTSTEP_LINK_LIBRARY glog gflags)

# add_subdirectory(src)
# add_subdirectory(third_party/pybind11)

# list(APPEND LIGHTSTEP_LINK_LIBRARY loader)

aux_source_directory (./src SRC_SO)
message(${SRC_SO})
set(appname ImageNet-Loader)
add_executable (${appname} "${SRC_SO}")
set_target_properties(${appname} PROPERTIES 
	CXX_STANDARD 17
	CXX_EXTENSIONS OFF
	DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
	)
target_link_libraries(${appname} ${LIGHTSTEP_LINK_LIBRARY})


aux_source_directory (./mysrc SRC_Files)
message(${SRC_Files})
set(appname MyImageLoader)
add_executable (${appname} "${SRC_Files}")
set_target_properties(${appname} PROPERTIES 
	CXX_STANDARD 17
	CXX_EXTENSIONS OFF
	DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
	)
target_link_libraries(${appname} ${LIGHTSTEP_LINK_LIBRARY})

# pybind11_add_module(dataloader SHARED wrapper.cpp)
# target_link_libraries(dataloader ${LIGHTSTEP_LINK_LIBRARY})
# set_target_properties(dataloader
    # PROPERTIES
    # LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/cdataloader)
