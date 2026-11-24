cmake_minimum_required(VERSION 3.24)
set(practice_cuda_arch "120a-real")
if(DEFINED ENV{PRACTICE_CUDA_ARCH})
  set(practice_cuda_arch "$ENV{PRACTICE_CUDA_ARCH}")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${practice_cuda_arch}" CACHE STRING "CUDA architectures")
project(cuda_operator_lab LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The practice launcher uses a separate build directory for each solution.
if(DEFINED CUDA_PRACTICE_SOURCE)
  add_library(practice_solution SHARED "${CUDA_PRACTICE_SOURCE}")
  set_target_properties(practice_solution PROPERTIES CUDA_ARCHITECTURES "${practice_cuda_arch}")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(practice_solution PRIVATE CUDA::cublas CUDA::cublasLt)
  target_include_directories(practice_solution SYSTEM PRIVATE references/cutlass/include)
  target_compile_options(practice_solution PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  return()
endif()

# Generate compilation flags that editors and language servers can use.
file(GLOB practice_sources CONFIGURE_DEPENDS
  "practice/leetgpu/*/solution.cu" "practice/xpuoj/*/solution.cu")
if(practice_sources)
  add_library(practice_editor_flags OBJECT EXCLUDE_FROM_ALL ${practice_sources})
  set_target_properties(practice_editor_flags PROPERTIES CUDA_ARCHITECTURES "${practice_cuda_arch}")
  target_include_directories(practice_editor_flags SYSTEM PRIVATE references/cutlass/include)
  target_compile_options(practice_editor_flags PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
