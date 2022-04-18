#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( _NVHPC_STDPAR_GPU )
  #include "Cuda_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
