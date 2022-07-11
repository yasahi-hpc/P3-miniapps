#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <vector>
#include <thrust/complex.h>
#include "../Iteration.hpp"
#include "View.hpp"

#if defined( ENABLE_CUDA ) || defined( ENABLE_HIP )
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_right;
  using default_iterate_layout = stdex::layout_right;
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
    #define SIMD_WIDTH 8
  #else
    #define SIMD_LOOP
    #define SIMD_WIDTH 1
  #endif
#endif

using float32 = float;
using float64 = double;
using complex64 = thrust::complex<float32>;
using complex128 = thrust::complex<float64>;

using Real = float64;
using size_type = uint64_t; // working with uint32_t

template <size_t ND>
using shape_nd = std::array<int, ND>;

namespace stdex = std::experimental;

template < typename ScalarType > 
using View1D = View<ScalarType, stdex::dextents< size_type, 1 >, default_layout >;
template < typename ScalarType > 
using View2D = View<ScalarType, stdex::dextents< size_type, 2 >, default_layout >;
template < typename ScalarType > 
using View3D = View<ScalarType, stdex::dextents< size_type, 3 >, default_layout >;
template < typename ScalarType > 
using View4D = View<ScalarType, stdex::dextents< size_type, 4 >, default_layout >;

using RealView1D = View1D<Real>;
using RealView2D = View2D<Real>;
using RealView3D = View3D<Real>;
using RealView4D = View4D<Real>;

using ComplexView1D = View1D< thrust::complex<Real> >;
using ComplexView2D = View2D< thrust::complex<Real> >;

template < size_t ND > using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
