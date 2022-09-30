# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("onnxinfer")

set(ONNX_LIB_DIR . )
# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_compile_options(
#  -DDEBUG_STFT
  -DDEBUG_FEAT
# -DDEBUG_WAV
# -DDEBUG_INF
  -mfma
  -mavx2
  -fno-permissive
  -Wno-deprecated-declarations
)

add_executable( # Sets the name of the library.
        onnxinfer

        # Provides a relative path to your source file(s).
        stfft.cpp
        wavread.cpp
        feat.cpp
        onnx_inference.cpp)


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.


set(ONNXRUNTIME_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.so)


target_compile_definitions(onnxinfer
        PRIVATE
        ${ORR_DEFINITIONS})


# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        onnxinfer

        ${ONNXRUNTIME_PATH}
)
