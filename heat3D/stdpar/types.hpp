#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <math.h>
#include <stdlib.h>
#include <vector>
#include <experimental/mdspan>
#include "View.hpp"
#include "../Iteration.hpp"

namespace stdex = std::experimental;

#if defined( _NVHPC_STDPAR_GPU )
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

template < size_t ND >
using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
