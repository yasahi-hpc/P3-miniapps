#ifndef __COMPLEX_TYPE_HPP__
#define __COMPLEX_TYPE_HPP__

namespace Impl {
  #if defined(ENABLE_KOKKOS)
    #include <Kokkos_Core.hpp>
    template <typename RealType>
    using complex = Kokkos::complex<RealType>;
  #elif defined(ENABLE_STDPAR) || defined( ENABLE_OPENMP_OFFLOAD )
    #include <complex>
    template <typename RealType>
    using complex = std::complex<RealType>;
  #elif defined(ENABLE_CUDA) || defined(ENABLE_HIP) || defined(ENABLE_THRUST)
    #include <thrust/complex.h>
    template <typename RealType>
    using complex = thrust::complex<RealType>;
  #else
    #include <complex>
    template <typename RealType>
    using complex = std::complex<RealType>;
  #endif
};

#endif
