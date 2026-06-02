cmake_minimum_required(VERSION 3.24)
project(int8_hybrid_moe_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Find PyTorch
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_CMAKE_PREFIX_PATH)
  message(FATAL_ERROR "Cannot find torch cmake_prefix_path. Please install PyTorch first.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# CUDA architectures
# ---------------------------------------------------------------------------
set(INT8HYBRID_CUDA_ARCHS "80;89" CACHE STRING "CUDA arch list for build")

# ---------------------------------------------------------------------------
# CUTLASS
# ---------------------------------------------------------------------------
if(NOT DEFINED INT8HYBRID_CUTLASS_DIR)
  if(DEFINED ENV{CUTLASS_DIR})
    set(INT8HYBRID_CUTLASS_DIR "$ENV{CUTLASS_DIR}")
  else()
    # Default: use the git submodule at include/cutlass
    set(INT8HYBRID_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/include/cutlass")
  endif()
endif()

if(NOT EXISTS "${INT8HYBRID_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "Cannot find CUTLASS headers. Set -DINT8HYBRID_CUTLASS_DIR=<cutlass_root> "
    "and ensure include/cutlass/cutlass.h exists.\n"
    "Tried: ${INT8HYBRID_CUTLASS_DIR}")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SRCS
  csrc/torch_bindings.cpp
  csrc/int8_grouped_gemm.cu
  csrc/int8_hybrid_grouped_gemm.cu
  csrc/moe_permute_unpermute_op.cu
  csrc/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu
  csrc/activation_kernels.cu
)

# ---------------------------------------------------------------------------
# Build shared library
# ---------------------------------------------------------------------------
add_library(int8_hybrid_moe_ops SHARED ${SRCS})

set_target_properties(int8_hybrid_moe_ops PROPERTIES
  CUDA_ARCHITECTURES "${INT8HYBRID_CUDA_ARCHS}"
  OUTPUT_NAME "int8_hybrid_moe_ops"
)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(int8_hybrid_moe_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_include_directories(int8_hybrid_moe_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${INT8HYBRID_CUTLASS_DIR}/include
)

target_link_libraries(int8_hybrid_moe_ops PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
)
