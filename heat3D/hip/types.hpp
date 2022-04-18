#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <math.h>
#include <stdlib.h>
#include <vector>
#include <experimental/mdspan>
#include <layout_contiguous/layout_contiguous.hpp>
#include "view.hpp"

using float64 = double;
using float32 = float;

using Real = float64;

using Index_t = int32_t;

namespace stdex = std::experimental;

template < class ScalarType >
using View1D = View<ScalarType, stdex::dextents< 1 >, layout_contiguous_at_left >;
template < class ScalarType >
using View2D = View<ScalarType, stdex::dextents< 2 >, layout_contiguous_at_left >;
template < class ScalarType >
using View3D = View<ScalarType, stdex::dextents< 3 >, layout_contiguous_at_left >;


#endif
