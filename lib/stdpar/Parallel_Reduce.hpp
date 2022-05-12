#ifndef __PARALLEL_REDUCE_HPP__
#define __PARALLEL_REDUCE_HPP__

#include <thrust/iterator/counting_iterator.h>
#include "../Iteration.hpp"

using counting_iterator = thrust::counting_iterator<int>;

namespace Impl {
  // Only for 1D case
  /*
  template <typename... I, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
  void for_each(const I... indices, const FunctorType f) {
    static_assert(sizeof...(I) <= 2, "range should be given in {start, stop} or {stop}");
    using index_type = std::tuple_element_t<0, std::tuple<I...>>;

    index_type start, stop;
    index_type indices_tmp[sizeof...(I)] = {indices...};

    if(sizeof...(I) == 1) {
      start = 0;
      stop = indices_tmp[0];
    } else {
      start = indices_tmp[0];
      stop  = indices_tmp[1];
    }
    index_type n = stop - start;

    std::for_each_n(std::execution::par_unseq,
                    counting_iterator(0), n,
                    [=](const int idx) {
                      const int i0 = idx + start;
                      f(i0);
                    });
  }
  */

  template <class IteratePolicy, class OutputType, class UnarayOperation, class BinaryOperation,
            std::enable_if_t<std::is_invocable_v< UnarayOperation, int >, std::nullptr_t> = nullptr>
  void transform_reduce(const IteratePolicy iterate_policy,
                        BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result
                       ) {
    static_assert( IteratePolicy::rank() == 1 );
    const auto start = iterate_policy.start();
    const auto start0 = start[0];
    auto n = iterate_policy.size();
    OutputType init = result;

    result = std::transform_reduce(std::execution::par_unseq,
                                   counting_iterator(0), counting_iterator(0)+n,
                                   init,
                                   binary_op,
                                   [=] (const int idx) {
                                     const int i0 = idx + start0;
                                     return unary_op(i0);
                                   }
                                  );
  }

  template <class IteratePolicy, class OutputType, class UnarayOperation, class BinaryOperation,
            std::enable_if_t<std::is_invocable_v< UnarayOperation, int, int >, std::nullptr_t> = nullptr>
  void transform_reduce(const IteratePolicy iterate_policy,
                        BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result
                       ) {
    static_assert( IteratePolicy::rank() == 2 );
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1];
    const auto n0 = strides[0], n1 = strides[1];
    auto n = iterate_policy.size();
    OutputType init = result;

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i0 = idx % n0 + start0;
                                       const int i1 = idx / n0 + start1;
                                       return unary_op(i0, i1);
                                     }
                                    );
    } else {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i0 = idx / n0 + start0;
                                       const int i1 = idx % n0 + start1;
                                       return unary_op(i0, i1);
                                     }
                                    );
    }
  }

  template <class IteratePolicy, class OutputType, class UnarayOperation, class BinaryOperation,
            std::enable_if_t<std::is_invocable_v< UnarayOperation, int, int, int >, std::nullptr_t> = nullptr>
  void transform_reduce(const IteratePolicy iterate_policy,
                        BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result
                       ) {
    static_assert( IteratePolicy::rank() == 3 );
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2];
    auto n = iterate_policy.size();
    OutputType init = result;

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i0  = idx % n0 + start0;
                                       const int i12 = idx / n0;
                                       const int i1  = i12%n1 + start1;
                                       const int i2  = i12/n1 + start2;
                                       return unary_op(i0, i1, i2);
                                     }
                                    );
    } else {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i2  = idx % n2 + start2;
                                       const int i01 = idx / n2;
                                       const int i1  = i01%n1 + start1;
                                       const int i0  = i01/n1 + start0;
                                       return unary_op(i0, i1, i2);
                                     }
                                    );
    }
  }

  template <class IteratePolicy, class OutputType, class UnarayOperation, class BinaryOperation,
            std::enable_if_t<std::is_invocable_v< UnarayOperation, int, int, int, int >, std::nullptr_t> = nullptr>
  void transform_reduce(const IteratePolicy iterate_policy,
                        BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result
                       ) {
    static_assert( IteratePolicy::rank() == 4 );
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2], start3 = start[3];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2], n3 = strides[3];
    auto n = iterate_policy.size();
    OutputType init = result;

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i0   = idx % n0 + start0;
                                       const int i123 = idx / n0;
                                       const int i1   = i123%n1 + start1;
                                       const int i23  = i123/n1;
                                       const int i2   = i23%n2 + start2;
                                       const int i3   = i23/n2 + start3;
                                       return unary_op(i0, i1, i2, i3);
                                     }
                                    );
    } else {
      result = std::transform_reduce(std::execution::par_unseq,
                                     counting_iterator(0), counting_iterator(0)+n,
                                     init,
                                     binary_op,
                                     [=] (const int idx) {
                                       const int i3   = idx % n3 + start3;
                                       const int i012 = idx / n3;
                                       const int i2   = i012%n2 + start2;
                                       const int i01  = i012/n2;
                                       const int i1   = i01%n1 + start1;
                                       const int i0   = i01/n1 + start0;
                                       return unary_op(i0, i1, i2, i3);
                                     }
                                    );
    }
  }

};

#endif
