#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_OffsetView.hpp>
#include <array>

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

using complex64 = Kokkos::complex<float32>;
using complex128 = Kokkos::complex<float64>;

template <unsigned ND> using coord_t = std::array<int, ND>;
template <unsigned ND> using shape_t = std::array<int, ND>;

typedef Kokkos::DefaultExecutionSpace execution_space;

#if defined(SIMD)
  #if defined(FUJI)
    #define LOOP_IVDEP _Pragma("loop simd")
    #define LOOP_SIMD _Pragma("loop simd")
  #else
    #define LOOP_IVDEP _Pragma("ivdep")
    #define LOOP_SIMD _Pragma("omp simd")
  #endif
#else
  #define LOOP_IVDEP
  #define LOOP_SIMD
#endif

#define LONG_BUFFER_WIDTH 256

#if defined ( LAYOUT_LEFT )
  // For layout optimization and layout left
  template <typename T> using View1D = Kokkos::View<T*, execution_space>;
  template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using View3D = Kokkos::View<T***, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using View4D = Kokkos::View<T****, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using View5D = Kokkos::View<T*****, Kokkos::LayoutLeft, execution_space>;
  
  template <typename T> using OffsetView1D = Kokkos::Experimental::OffsetView<T*, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using OffsetView2D = Kokkos::Experimental::OffsetView<T**, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using OffsetView3D = Kokkos::Experimental::OffsetView<T***, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using OffsetView4D = Kokkos::Experimental::OffsetView<T****, Kokkos::LayoutLeft, execution_space>;
  template <typename T> using OffsetView5D = Kokkos::Experimental::OffsetView<T*****, Kokkos::LayoutLeft, execution_space>;

  // Range Policies
  template < size_t DIM > using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Left, Kokkos::Iterate::Left> >;
#else
  template <typename T> using View1D = Kokkos::View<T*, execution_space>;
  template <typename T> using View2D = Kokkos::View<T**, execution_space>;
  template <typename T> using View3D = Kokkos::View<T***, execution_space>;
  template <typename T> using View4D = Kokkos::View<T****, execution_space>;
  template <typename T> using View5D = Kokkos::View<T*****, execution_space>;
  
  template <typename T> using OffsetView1D = Kokkos::Experimental::OffsetView<T*, execution_space>;
  template <typename T> using OffsetView2D = Kokkos::Experimental::OffsetView<T**, execution_space>;
  template <typename T> using OffsetView3D = Kokkos::Experimental::OffsetView<T***, execution_space>;
  template <typename T> using OffsetView4D = Kokkos::Experimental::OffsetView<T****, execution_space>;
  template <typename T> using OffsetView5D = Kokkos::Experimental::OffsetView<T*****, execution_space>;

  // Range Policies
  template < size_t DIM > using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
#endif
  
  using RealView1D = View1D<float64>;
  using RealView2D = View2D<float64>;
  using RealView3D = View3D<float64>;
  using RealView4D = View4D<float64>;

  using RealOffsetView1D = OffsetView1D<float64>;
  using RealOffsetView2D = OffsetView2D<float64>;
  using RealOffsetView3D = OffsetView3D<float64>;
  using RealOffsetView4D = OffsetView4D<float64>;

  using ComplexView1D = View1D<complex128>;
  using ComplexView2D = View2D<complex128>;

  using IntView1D = View1D<int>;
  using IntView2D = View2D<int>;

#endif
