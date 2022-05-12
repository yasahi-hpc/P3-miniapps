#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <complex>
#include <omp.h>
#include <experimental/mdspan>
#include "OpenACC_View.hpp"

namespace stdex = std::experimental;

// Directives to force vectorization
#if defined ( ENABLE_OPENACC )
  using default_layout = stdex::layout_left;
  #define LOOP_SIMD _Pragma("acc loop vector independent")
  #define SIMD_WIDTH 1
#else
  using default_layout = stdex::layout_right;
  #define SIMD_WIDTH 8
  
  #if defined(SIMD)
    #if defined(FUJI)
      #define LOOP_SIMD _Pragma("loop simd")
    #else
      #define LOOP_SIMD _Pragma("omp simd")
    #endif
  #else
    #define LOOP_SIMD
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
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
template <typename RealType> using Complex = std::complex<RealType>;

const complex128 I = complex128(0., 1.);

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

using IntView1D = View1D<int>;
using IntView2D = View2D<int>;

#endif
