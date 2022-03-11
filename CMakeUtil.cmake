include(CMakeParseArguments)

function(target_do_force_link_libraries target visibility lib)
    if(MSVC)
        target_link_libraries(${target} ${visibility} "/WHOLEARCHIVE:${lib}")
    elseif(APPLE)
        target_link_libraries(${target} ${visibility} -Wl,-force_load ${lib})
    else()
        target_link_libraries(${target} ${visibility} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
    endif()
endfunction()

function(target_force_link_libraries target)
    cmake_parse_arguments(FLINK
            ""
            ""
            "PUBLIC;INTERFACE;PRIVATE"
            ${ARGN}
            )

    foreach(lib IN LISTS FLINK_PUBLIC)
        target_do_force_link_libraries(${target} PUBLIC ${lib})
    endforeach()

    foreach(lib IN LISTS FLINK_INTERFACE)
        target_do_force_link_libraries(${target} INTERFACE ${lib})
    endforeach()

    foreach(lib IN LISTS FLINK_PRIVATE)
        target_do_force_link_libraries(${target} PRIVATE ${lib})
    endforeach()
endfunction()

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