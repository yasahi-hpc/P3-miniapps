#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( ENABLE_CUDA )
  #include "../Cuda_FFT.hpp"
#elif defined( ENABLE_HIP )
  #include "../HIP_FFT.hpp"
#else
  #include "../OpenMP_FFT.hpp"
#endif

#endif
