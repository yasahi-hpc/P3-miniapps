#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <math.h>
#include <stdlib.h>
#include <vector>
#include <layout_contiguous/layout_contiguous.hpp>
#include <complex>
#include<omp.h>
#include "OpenMP_View.hpp"

#if defined( ENABLE_OPENMP_OFFLOAD )
  using default_layout = layout_contiguous_at_left;
  #define LOOP_SIMD
  #define SIMD_WIDTH 1
#else
  using default_layout = layout_contiguous_at_left;
  struct int1 {int x;};
  struct int2 {int x, y;};
  struct int3 {int x, y, z;};
  struct int4 {int x, y, z, w;};
  static inline int1 make_int1(int x) {int1 t; t.x=x; return t;}
  static inline int2 make_int2(int x, int y) {int2 t; t.x=x; t.y=y; return t;}
  static inline int3 make_int3(int x, int y, int z) {int3 t; t.x=x; t.y=y; t.z=z; return t;}
  static inline int4 make_int4(int x, int y, int z, int w) {int4 t; t.x=x; t.y=y; t.z=z; t.w=w; return t;}
  #if defined(SIMD)
    #if defined(FUJI)
      #define LOOP_SIMD _Pragma("loop simd")
    #else
      #define LOOP_SIMD _Pragma("omp simd")
    #endif
    #define SIMD_WIDTH 8
  #else
    #define LOOP_SIMD
    #define SIMD_WIDTH 1
  #endif
#endif

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
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
template <typename RealType> using Complex = std::complex<RealType>;

const complex128 I = complex128(0., 1.);

using Real = float64;

template < typename ScalarType >
using View1D = View<ScalarType, 1, default_layout >;
template < typename ScalarType >
using View2D = View<ScalarType, 2, default_layout >;
template < typename ScalarType >
using View3D = View<ScalarType, 3, default_layout >;
template < typename ScalarType >
using View4D = View<ScalarType, 4, default_layout >;

// RealView
using RealView1D = View1D<float64>;
using RealView2D = View2D<float64>;
using RealView3D = View3D<float64>;
using RealView4D = View4D<float64>;

using ComplexView1D = View1D<complex128>;
using ComplexView2D = View2D<complex128>;

#endif
