#!/bin/bash

export PATH=/opt/python/cp37-cp37m/bin:$PATH
export BUILD_TYPE=Release

pip install conan==1.58 ninja
cmake -E make_directory build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE -Donnxruntime_BUILD_UNIT_TESTS=OFF -DBUILD_TEST_EXE=OFF
cmake --build . --config $BUILD_TYPE
cmake --install . --prefix ../install
