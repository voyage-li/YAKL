cmake_minimum_required(VERSION 3.16)

project(stencil CXX)

enable_language(CXX)
enable_language(C)

set(CMAKE_CXX_STANDARD 17)

set(Kokkos_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../../kokkos)
set(Kokkos_BIN  ${CMAKE_CURRENT_BINARY_DIR}/kokkos)
add_subdirectory(${Kokkos_HOME} ${Kokkos_BIN})
include_directories(${Kokkos_INCLUDE_DIRS_RET})

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

add_executable(stencil stencil.cpp)

if(Kokkos_ENABLE_SW64)
	set_source_files_properties(stencil stencil.cpp PROPERTIES COMPILE_OPTIONS "-mhybrid-coding;-fpermissive")
	target_compile_definitions(stencil PUBLIC Kokkos_ENABLE_SW64)
endif()

if(YAKL_PROFILE)
	target_compile_definitions(stencil PUBLIC YAKL_PROFILE)
endif()

target_link_libraries(stencil yakl kokkos)