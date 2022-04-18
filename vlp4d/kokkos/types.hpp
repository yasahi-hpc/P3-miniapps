#ifndef __TYPES_H__
#define __TYPES_H__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <array>
using execution_space = Kokkos::DefaultExecutionSpace;

// New directive for vectorization with OpenMP background
#if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
  #define KOKKOS_SIMD
#else
  #define KOKKOS_SIMD _Pragma("omp simd")
#endif

using float32 = float;
using float64 = double;
using complex64 = Kokkos::complex<float32>;
using complex128 = Kokkos::complex<float64>;

// Multidimensional view types
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
template <typename T> using View2D = Kokkos::View<T**, execution_space>;
template <typename T> using View3D = Kokkos::View<T***, execution_space>;
template <typename T> using View4D = Kokkos::View<T****, execution_space>;
template <typename T> using View5D = Kokkos::View<T*****, execution_space>;

using RealView1D = View1D<float64>;
using RealView2D = View2D<float64>;
using RealView3D = View3D<float64>;
using RealView4D = View4D<float64>;

using ComplexView1D = View1D<complex128>;
using ComplexView2D = View2D<complex128>;

template < size_t DIM > using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;

template <size_t DIM> using coord_t = std::array<int, DIM>;
template <size_t DIM> using shape_t = std::array<int, DIM>;

struct double_pair {
  double x, y;
  KOKKOS_INLINE_FUNCTION
  double_pair(double xinit, double yinit) 
    : x(xinit), y(yinit) {};

  KOKKOS_INLINE_FUNCTION
  double_pair()
    : x(0.), y(0.) {};

  KOKKOS_INLINE_FUNCTION
  double_pair& operator += (const double_pair& src) {
    x += src.x;
    y += src.y;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile double_pair& operator += (const volatile double_pair& src) volatile {
    x += src.x;
    y += src.y;
    return *this;
  }
};

#if ! ( defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP ) )
struct int2 {
  int x, y;
};

KOKKOS_INLINE_FUNCTION
int2 make_int2(int x, int y) {
  int2 t; t.x = x; t.y= y; return t;
};

struct int3 {
  int x, y, z;
};

KOKKOS_INLINE_FUNCTION
int3 make_int3(int x, int y, int z) {
  int3 t; t.x = x; t.y = y; t.z = z; return t;
};

struct int4 {
  int x, y, z, w;
};

KOKKOS_INLINE_FUNCTION
int4 make_int4(int x, int y, int z, int w) {
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
};
#endif

#endif
