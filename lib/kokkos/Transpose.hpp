#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined( KOKKOS_ENABLE_CUDA )
  //#include "Kokkos_Cuda_Transpose.hpp"
  #include "../Cuda_Transpose.hpp"
#elif defined( KOKKOS_ENABLE_HIP )
  //#include "Kokkos_HIP_Transpose.hpp"
  #include "../HIP_Transpose.hpp"
#else
  //#include "Kokkos_OpenMP_Transpose.hpp"
  #include "../OpenMP_Transpose.hpp"
#endif

#endif
