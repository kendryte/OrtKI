cmake_minimum_required(VERSION 3.13)

project(OrtKI)

set(SRCS src/main.cpp
        src/op_executor.cpp
        src/allocator_manager.cpp
        src/default_providers.cpp
        src/environment.cpp)

add_library(ort_ki SHARED ${SRCS})

add_compile_definitions(ONNX_ML=1)
add_compile_definitions(ONNX_NAMESPACE=onnx)
set(onnxruntime_BUILD_SHARED_LIB ON)

add_subdirectory(onnxruntime/cmake/)
message(STATUS "msg")

set(INCLUDE_DIRS
        include
        onnxruntime/cmake/external/onnx
        onnxruntime/onnxruntime/core/providers
        onnxruntime/onnxruntime/
        onnxruntime/include/onnxruntime
        onnxruntime/cmake/external/mp11/include
        onnxruntime/cmake/external/nsync/public
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/cmake/external/onnx
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/cmake)

target_include_directories(ort_ki PRIVATE ${INCLUDE_DIRS})

target_link_libraries(ort_ki
        onnxruntime
        onnx
        libprotoc)

#add_executable(main main.cpp)
#target_include_directories(main PRIVATE ${INCLUDE_DIRS})
#target_link_libraries(main ort_ki onnxruntime onnx libprotoc)