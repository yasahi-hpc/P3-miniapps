#ifndef __LAYOUT_HPP__
#define __LAYOUT_HPP__

#if defined(ENABLE_KOKKOS)
  #include <Kokkos_Core.hpp>
  using layout_left  = Kokkos::LayoutLeft;
  using layout_right = Kokkos::LayoutRight;
#else
  #include <experimental/mdspan>
  namespace stdex = std::experimental;
  using layout_left = stdex::layout_left;
  using layout_right = stdex::layout_right;
#endif

#endif
