#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( KOKKOS_ENABLE_CUDA )
  //#include "Kokkos_Cuda_FFT.hpp"
  #include "../Cuda_FFT.hpp"
#elif defined( KOKKOS_ENABLE_HIP )
  //#include "Kokkos_HIP_FFT.hpp"
  #include "../HIP_FFT.hpp"
#else
  //#include "Kokkos_OpenMP_FFT.hpp"
  #include "../OpenMP_FFT.hpp"
#endif

#endif
