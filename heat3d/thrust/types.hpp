#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <math.h>
#include <stdlib.h>
#include <vector>
#include <thrust/complex.h>
#include <experimental/mdspan>
#include "View.hpp"
#include "../Iteration.hpp"

namespace stdex = std::experimental;

#if defined( ENABLE_CUDA ) || defined( ENABLE_HIP )
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_right;
  using default_iterate_layout = stdex::layout_right;
  
  #define SIMD_WIDTH 8
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
  #else
    #define SIMD_LOOP
  #endif

  /*
  struct int1 {int x;};
  struct int2 {int x, y;};
  struct int3 {int x, y, z;};
  struct int4 {int x, y, z, w;};
  static inline int1 make_int1(int x) {int1 t; t.x=x; return t;}
  static inline int2 make_int2(int x, int y) {int2 t; t.x=x; t.y=y; return t;}
  static inline int3 make_int3(int x, int y, int z) {int3 t; t.x=x; t.y=y; t.z=z; return t;}
  static inline int4 make_int4(int x, int y, int z, int w) {int4 t; t.x=x; t.y=y; t.z=z; t.w=w; return t;}
  */
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

using Real = float64;

template < class ScalarType >
using View1D = View<ScalarType, stdex::dextents< 1 >, default_layout >;
template < class ScalarType >
using View2D = View<ScalarType, stdex::dextents< 2 >, default_layout >;
template < class ScalarType >
using View3D = View<ScalarType, stdex::dextents< 3 >, default_layout >;

using RealView1D = View1D<Real>;
using RealView2D = View2D<Real>;
using RealView3D = View3D<Real>;

template < size_t ND >
using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
