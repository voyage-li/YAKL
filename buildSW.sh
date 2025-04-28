#!/bin/bash

cd "${0%/*}" || exit
BUILD_DIR=objdir

module load gcc/8.3.1
module load swgcc/1449
module load mpi/swuc

if [ ! -e "${BUILD_DIR}/CMakeCache.txt" ]; then
  cmake -S ./example/build_swuc_example -B${BUILD_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_SW64=ON \
        -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
        --toolchain $(pwd)/swuc.cmake
fi

cmake --build ${BUILD_DIR} "$@"