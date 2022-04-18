#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <Kokkos_Core.hpp>

namespace Impl {

  template <class ViewType>
  void swap(ViewType &a, ViewType &b) {
    ViewType tmp = a;
    a = b;
    b = tmp;
  }

  template <class ViewType>
  struct DeepCopy_functor {
    typedef typename ViewType::value_type value_type;
    ViewType   a_;
    ViewType   b_;
    value_type *ptr_a_;
    value_type *ptr_b_;

    DeepCopy_functor(ViewType &a, ViewType &b)
      : a_(a), b_(b) {
      ptr_a_ = a_.data();
      ptr_b_ = b_.data();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      ptr_a_[i] = ptr_b_[i];
    }
  };

  template <class ViewType>
  struct DeepCopies_functor {
    typedef typename ViewType::value_type value_type;
    ViewType   a_;
    ViewType   b_;
    ViewType   c_;
    value_type *ptr_a_;
    value_type *ptr_b_;
    value_type *ptr_c_;

    DeepCopies_functor(ViewType &a, ViewType &b, ViewType &c)
      : a_(a), b_(b), c_(c) {
      ptr_a_ = a_.data();
      ptr_b_ = b_.data();
      ptr_c_ = c_.data();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      ptr_a_[i] = ptr_b_[i] = ptr_c_[i];
    }
  };

  template <class ViewType>
  void deep_copy(ViewType &a, ViewType &b) {
    const size_t n = a.size();
    Kokkos::parallel_for("deep_copy", n, DeepCopy_functor<ViewType>(a, b));
  }

  template <class ViewType>
  void deep_copy(ViewType &a, ViewType &b, ViewType &c) {
    const size_t n = a.size();
    Kokkos::parallel_for("deep_copy", n, DeepCopies_functor<ViewType>(a, b, c));
  }

  template <class ViewType>
  void free(ViewType &a) {
    a = ViewType();
  }

  template <class ViewType, typename ScalarType>
  void fill(ViewType &a, ScalarType value) {
    const size_t n = a.size();
    typedef typename ViewType::value_type value_type;
    Kokkos::parallel_for("fill", n, KOKKOS_LAMBDA(const int i) {
      a.data()[i] = static_cast<value_type>(value);
    });
  }
};

#endif
