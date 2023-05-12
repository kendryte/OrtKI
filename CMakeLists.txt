cmake_minimum_required(VERSION 3.13)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(OrtKI)

include(ExternalProject)

include(cmake/configure-conan.cmake)
include(cmake/conan.cmake)
conan_cmake_run(CONANFILE conanfile.py
                BASIC_SETUP
                OPTIONS ${CONAN_OPTS}
                SETTINGS ${CONAN_SETTINGS}
                BUILD missing)

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
endif()

add_subdirectory(onnxruntime/cmake/)

set(INCLUDE_DIRS
        include
        onnxruntime/cmake/external/onnx
        onnxruntime/onnxruntime/core/providers
        onnxruntime/onnxruntime/
        onnxruntime/include/onnxruntime
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/cmake/external/onnx
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/abseil_cpp-src/
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/mp11-src/include
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/google_nsync-src/public
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/cmake)

target_include_directories(ortki PUBLIC ${INCLUDE_DIRS})

set(ONNXRUNTIME_TARGETS onnxruntime_common onnxruntime_graph onnxruntime_framework onnxruntime_util onnxruntime_providers onnxruntime_optimizer onnxruntime_session onnxruntime_mlas onnxruntime_flatbuffers)

# used for debug
function(get_all_targets var)
    set(targets)
    get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()
get_all_targets(all_targets)
message("All targets: ${all_targets}")


if(MSVC)
    target_link_libraries(ortki ${ONNXRUNTIME_TARGETS})
    target_link_options(ortki PUBLIC /WHOLEARCHIVE:$<TARGET_FILE:onnxruntime_providers>)
elseif(APPLE)
    target_link_libraries(ortki -Wl,-force_load ${ONNXRUNTIME_TARGETS})
else()
    target_link_libraries(ortki -Wl,--whole-archive ${ONNXRUNTIME_TARGETS} -Wl,--no-whole-archive -static-libstdc++ -Wl,--no-undefined)
endif()

target_link_libraries(ortki re2 onnx cpuinfo)

if(MSVC)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        target_link_libraries(ortki Dbghelp)
    endif()
else()
    target_link_libraries(ortki
            dl
            nsync_cpp)
endif()


if(BUILD_TEST_EXE)
    add_executable(run src/main.cpp)
    target_link_libraries(run ortki)
endif()

install(TARGETS ortki
        EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
