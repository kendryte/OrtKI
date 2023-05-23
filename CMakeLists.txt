cmake_minimum_required(VERSION 3.13)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(OrtKI)

if (NOT CONAN_CXX_FLAGS) 
    include(cmake/configure-conan.cmake)
    include(cmake/conan.cmake)
    conan_cmake_run(CONANFILE conanfile.py
                    BASIC_SETUP
                    OPTIONS ${CONAN_OPTS}
                    SETTINGS ${CONAN_SETTINGS}
                    BUILD missing)
endif()

find_package(onnxruntime REQUIRED CONFIG)

set(SRCS
        src/op_executor.cpp
        src/allocator_manager.cpp
        src/default_providers.cpp
        src/environment.cpp
        src/c_api.cpp
        src/operators.cpp
        src/operators_patch.cpp
        src/util.cpp)

add_library(ortki SHARED ${SRCS})

#add_compile_definitions(ONNX_ML=1)
#add_compile_definitions(ONNX_NAMESPACE=onnx)

set(onnxruntime_DISABLE_RTTI OFF)
set(onnxruntime_BUILD_UNIT_TESTS OFF)
set(onnxruntime_DISABLE_CONTRIB_OPS ON)
set(onnxruntime_DISABLE_ML_OPS ON)
set(onnxruntime_DISABLE_SPARSE_TENSORS OFF)
set(onnxruntime_DISABLE_OPTIONAL_TYPE ON)
set(protobuf_MSVC_STATIC_RUNTIME OFF)

if(MSVC)
    add_compile_options(/wd4127)
else()
    target_link_options(ortki PRIVATE -Wl,--no-undefined)
endif()

set(INCLUDE_DIRS
        include)

target_include_directories(ortki PUBLIC ${INCLUDE_DIRS})
target_include_directories(ortki PRIVATE ${onnxruntime_INCLUDE_DIR}/onnxruntime)
target_link_libraries(ortki PRIVATE onnxruntime::onnxruntime)

if (MSVC)
elseif(APPLE)
else()
    target_link_options(ortki PRIVATE -static-libstdc++)
endif()


if(BUILD_TEST_EXE)
    add_executable(run src/main.cpp)
    target_link_libraries(run PRIVATE ortki)
endif()

install(TARGETS ortki
        EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
    DIRECTORY ${INCLUDE_DIRS}/
    DESTINATION include/ortki
    FILES_MATCHING PATTERN "*.h*")