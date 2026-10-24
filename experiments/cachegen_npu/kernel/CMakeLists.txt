cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
project(cachegen_entropy_restore LANGUAGES ASC AICPU CXX)
add_library(cachegen_entropy_restore SHARED decode.aicpu launcher.asc)
set_target_properties(cachegen_entropy_restore PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(cachegen_entropy_restore PRIVATE
 $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
 $<$<COMPILE_LANGUAGE:AICPU>:-march=armv8-a+crypto>)
