#ifndef __PARALLEL_FOR_HPP__
#define __PARALLEL_FOR_HPP__

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <layout_contiguous/layout_contiguous.hpp>
#include "../Iteration.hpp"

using counting_iterator = thrust::counting_iterator<size_t>;

namespace Impl {
  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
  void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 1 );
    const auto start = iterate_policy.start();
    const auto start0 = start[0];
    auto n = iterate_policy.size();

    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                       const int i0 = idx + start0;
                       f(i0);
                     });
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr>
  void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 2 );
    //structured binding cannot be captured
    //auto [start0, start1] = iterate_policy.starts();
    //auto [n0, n1] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1];
    const auto n0 = strides[0], n1 = strides[1];
    auto n = iterate_policy.size();

    /*
    if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0 = idx % n0 + start0;
                         const int i1 = idx / n0 + start1;
                         f(i0, i1);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0 = idx / n0 + start0;
                         const int i1 = idx % n0 + start1;
                         f(i0, i1);
                       });
    }
    */
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr>
  void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 3 );
    //structured binding cannot be captured
    //auto [start0, start1, start2] = iterate_policy.start();
    //auto [n0, n1, n2] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2];
    auto n = iterate_policy.size();

    /*
    if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0  = idx % n0 + start0;
                         const int i12 = idx / n0;
                         const int i1  = i12%n1 + start1;
                         const int i2  = i12/n1 + start2;
                         f(i0, i1, i2);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i2  = idx % n2 + start2;
                         const int i01 = idx / n2;
                         const int i1  = i01%n1 + start1;
                         const int i0  = i01/n1 + start0;
                         f(i0, i1, i2);
                       });
    }
    */
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr>
  void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 4 );
    //structured binding cannot be captured
    //auto [start0, start1, start2, start3] = iterate_policy.starts();
    //auto [n0, n1, n2, n3] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2], start3 = start[3];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2], n3 = strides[3];
    auto n = iterate_policy.size();

    /*
    if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0   = idx % n0 + start0;
                         const int i123 = idx / n0;
                         const int i1   = i123%n1 + start1;
                         const int i23  = i123/n1;
                         const int i2   = i23%n2 + start2;
                         const int i3   = i23/n2 + start3;
                         f(i0, i1, i2, i3);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i3   = idx % n3 + start3;
                         const int i012 = idx / n3;
                         const int i2   = i012%n2 + start2;
                         const int i01  = i012/n2;
                         const int i1   = i01%n1 + start1;
                         const int i0   = i01/n1 + start0;
                         f(i0, i1, i2, i3);
                       });
    }
    */
  }
};

#endif
