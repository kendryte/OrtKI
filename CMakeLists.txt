cmake_minimum_required(VERSION 3.13)

project(OrtKI)

add_subdirectory(onnxruntime/cmake/)

set(SRCS main.cpp provider_test_utils.cc)
add_executable(ort_cpi ${SRCS})
## add common libraries
## add include
target_include_directories(ort_cpi PRIVATE
        include
        onnxruntime/onnxruntime/core/providers
        onnxruntime/onnxruntime/
        onnxruntime/include/onnxruntime)