#ifndef __PARALLEL_FOR_HPP__
#define __PARALLEL_FOR_HPP__

#if defined( ENABLE_CUDA ) && ! defined( ENABLE_THRUST )
  #include "Cuda_Parallel_For.hpp"
#elif defined( ENABLE_HIP ) && ! defined( ENABLE_THRUST )
  #include "HIP_Parallel_For.hpp"
#elif defined( ENABLE_OPENMP ) && ! defined( ENABLE_THRUST )
  #include "Openmp_Parallel_For.hpp"
#else
  #include "Thrust_Parallel_For.hpp"
#endif

#endif
