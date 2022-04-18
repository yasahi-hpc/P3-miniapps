#ifndef __UTILS_HPP__
#define __UTILS_HPP__

namespace Impl {
  template <class ViewType>
  void swap(ViewType &a, ViewType &b) {
    ViewType tmp = a;
    a = b;
    b = tmp;
  }
};

#endif
