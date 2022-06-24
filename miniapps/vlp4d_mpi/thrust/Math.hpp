#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace Impl {
  template <class ViewType>
  void swap(ViewType &a, ViewType &b) {
    b.swap(a);
  }

  template <class ViewType>
  void deep_copy(ViewType &a, const ViewType &b) {
    thrust::copy(thrust::device, b.data(), b.data()+b.size(), a.data());
  }
};

#endif
