#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined( ENABLE_CUDA )
  #include "OpenMP_Cuda_Transpose.hpp"
#elif defined( ENABLE_HIP )
  #include "OpenMP_HIP_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

#endif
