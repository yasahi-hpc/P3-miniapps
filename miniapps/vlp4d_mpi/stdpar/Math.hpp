#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <omp.h>
#include <execution>
#include <numeric>
#include <algorithm>

namespace Impl {
  template <class ViewType>
  void swap(ViewType &a, ViewType &b) {
    b.swap(a);
  }

  template <class ViewType>
  void deep_copy(ViewType &a, ViewType &b) {
    std::copy(std::execution::par_unseq, b.data(), b.data()+b.size(), a.data());
  }
};

#endif
