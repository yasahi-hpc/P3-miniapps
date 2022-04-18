#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( ENABLE_CUDA )
  #include "OpenMP_Cuda_FFT.hpp"
#elif defined( ENABLE_HIP )
  #include "OpenMP_HIP_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
