#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined( _NVHPC_STDPAR_GPU )
  #include "Cuda_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

#endif
