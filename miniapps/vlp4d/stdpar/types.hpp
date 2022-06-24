#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <vector>
#include <complex>
#include "../Iteration.hpp"
#include "View.hpp"

namespace stdex = std::experimental;

#if defined( _NVHPC_STDPAR_GPU )
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
#else
  struct uint1 {int x;};
  struct uint2 {int x, y;};
  struct uint3 {int x, y, z;};
  struct uint4 {int x, y, z, w;};
  static inline uint1 make_uint1(int x) {uint1 t; t.x=x; return t;}
  static inline uint2 make_uint2(int x, int y) {uint2 t; t.x=x; t.y=y; return t;}
  static inline uint3 make_uint3(int x, int y, int z) {uint3 t; t.x=x; t.y=y; t.z=z; return t;}
  static inline uint4 make_uint4(int x, int y, int z, int w) {uint4 t; t.x=x; t.y=y; t.z=z; t.w=w; return t;}
  using default_layout = stdex::layout_right;
  using default_iterate_layout = stdex::layout_right;
  #define SIMD_WIDTH 8
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
  #else
    #define SIMD_LOOP
  #endif
#endif

using float32 = float;
using float64 = double;
using complex64 = std::complex<float32>;
using complex128 = std::complex<float64>;

using Real = float64;
using size_type = uint32_t;

template <size_t ND>
using shape_nd = std::array<int, ND>;

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

using ComplexView1D = View1D< std::complex<Real> >;
using ComplexView2D = View2D< std::complex<Real> >;

template < size_t ND > using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
