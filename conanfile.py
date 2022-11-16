# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from conans import ConanFile, CMake, tools


class ortkiConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_find_package", "cmake_paths"
    options = {
        "shared": [True, False],
        "fPIC": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True
    }

    def requirements(self):
        pass

    def build_requirements(self):
        pass

    def configure(self):
        min_cppstd = "17"
        tools.check_min_cppstd(self, min_cppstd)

    def cmake_configure(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_CXX_STANDARD"] = 17
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.cmake_configure()
        cmake.build()
