#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <vector>
#include <thrust/complex.h>
#include "../Iteration.hpp"
#include "View.hpp"

namespace stdex = std::experimental;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
  #define SIMD_WIDTH 8
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
  #else
    #define SIMD_LOOP
  #endif
#endif

#define LONG_BUFFER_WIDTH 256

using int8  = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8  = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using float32 = float;
using float64 = double;
using complex64 = thrust::complex<float32>;
using complex128 = thrust::complex<float64>;

using Real = float64;

template <size_t ND>
using shape_nd = std::array<int, ND>;

template < typename ScalarType > 
using View1D = View<ScalarType, stdex::dextents< 1 >, default_layout >;
template < typename ScalarType > 
using View2D = View<ScalarType, stdex::dextents< 2 >, default_layout >;
template < typename ScalarType > 
using View3D = View<ScalarType, stdex::dextents< 3 >, default_layout >;
template < typename ScalarType > 
using View4D = View<ScalarType, stdex::dextents< 4 >, default_layout >;

using RealView1D = View1D<Real>;
using RealView2D = View2D<Real>;
using RealView3D = View3D<Real>;
using RealView4D = View4D<Real>;

using ComplexView1D = View1D< thrust::complex<Real> >;
using ComplexView2D = View2D< thrust::complex<Real> >;

using IntView1D = View1D<int>;
using IntView2D = View2D<int>;

template < size_t ND >
using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
