#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <vector>
#include <Kokkos_Core.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;

using float64 = double;
using float32 = float;

using Real = float64;

template < class ScalarType >
using View1D = Kokkos::View<ScalarType*, execution_space>;
template < class ScalarType >
using View2D = Kokkos::View<ScalarType**, execution_space>;
template < class ScalarType >
using View3D = Kokkos::View<ScalarType***, execution_space>;

template < size_t DIM > using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;

#endif
