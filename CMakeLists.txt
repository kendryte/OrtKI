cmake_minimum_required(VERSION 3.13)

project(OrtKI)

set(CMAKE_CXX_STANDARD 20)
if(MSVC)
    message(STATUS"cmake into msvc")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        message(STATUS"MSVC into Debug /MDd")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MDd")
    else()
        message(STATUS"MSVC into Release /MD")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD")
    endif()
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
endif()
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")

set(SRCS
        src/op_executor.cpp
        src/allocator_manager.cpp
        src/default_providers.cpp
        src/environment.cpp
        src/c_api.cpp
        src/operators.cpp
        src/operators_patch.cpp)

add_library(ortki SHARED ${SRCS})

add_compile_definitions(ONNX_ML=1)
add_compile_definitions(ONNX_NAMESPACE=onnx)

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
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/abseil_cpp-src/
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
    target_link_libraries(ortki -Wl,--whole-archive ${ONNXRUNTIME_TARGETS} -Wl,--no-whole-archive)
endif()

if(MSVC)
    target_link_libraries(ortki
            absl_raw_hash_set
            absl_hash_function_defaults
            re2
            onnx)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        target_link_libraries(ortki Dbghelp)
    endif()
else()
    target_link_libraries(ortki
            nsync_cpp
            dl
            absl_raw_hash_set
            absl_hash_function_defaults
            re2
            onnx)
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
