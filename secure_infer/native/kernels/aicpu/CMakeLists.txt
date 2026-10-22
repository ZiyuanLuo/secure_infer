cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
project(runtime_aicpu_ghash LANGUAGES ASC AICPU CXX)

set(AICPU_AES_INTERLEAVE "4" CACHE STRING "AICPU AES-CTR interleave factor")
set(AICPU_GHASH_FOLD "8" CACHE STRING "AICPU GHASH fold factor")
set(AICPU_GHASH_MULTICHAIN "0" CACHE STRING "AICPU GHASH multi-chain factor")

add_library(runtime_aicpu_ghash SHARED ghash.aicpu launcher.asc)
set_target_properties(runtime_aicpu_ghash PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(
  runtime_aicpu_ghash PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
  $<$<COMPILE_LANGUAGE:AICPU>:-march=armv8-a+crypto>)
target_compile_definitions(
  runtime_aicpu_ghash PRIVATE
  PIPE_AICPU_AES_INTERLEAVE=${AICPU_AES_INTERLEAVE}
  PIPE_AICPU_GHASH_FOLD=${AICPU_GHASH_FOLD}
  PIPE_AICPU_GHASH_MULTICHAIN=${AICPU_GHASH_MULTICHAIN})
target_link_libraries(runtime_aicpu_ghash PRIVATE crypto dl)
