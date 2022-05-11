#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_OffsetView.hpp>
#include <array>

// Directives to force vectorization
#if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
  #define LOOP_SIMD
  #define SIMD_WIDTH 1
#else
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

using execution_space = Kokkos::DefaultExecutionSpace;

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
using complex64 = Kokkos::complex<float>;
using complex128 = Kokkos::complex<double>;

const complex128 I = complex128(0., 1.);

template < typename ScalarType >
using View1D = Kokkos::View<ScalarType*, execution_space>;
template < typename ScalarType >
using View2D = Kokkos::View<ScalarType**, execution_space>;
template < typename ScalarType >
using View3D = Kokkos::View<ScalarType***, execution_space>;

template < typename ScalarType >
using OffsetView1D = Kokkos::Experimental::OffsetView<ScalarType*, execution_space>;
template < typename ScalarType >
using OffsetView2D = Kokkos::Experimental::OffsetView<ScalarType**, execution_space>;
template < typename ScalarType >
using OffsetView3D = Kokkos::Experimental::OffsetView<ScalarType***, execution_space>;

// RealView
using RealView1D = View1D<float64>;
using RealView2D = View2D<float64>;
using RealView3D = View3D<float64>;

using RealOffsetView1D = OffsetView1D<float64>;
using RealOffsetView2D = OffsetView2D<float64>;
using RealOffsetView3D = OffsetView3D<float64>;

template < size_t DIM > using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;

#endif
