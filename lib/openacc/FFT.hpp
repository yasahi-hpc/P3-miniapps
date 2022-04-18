#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( ENABLE_OPENACC )
  #include "OpenACC_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
