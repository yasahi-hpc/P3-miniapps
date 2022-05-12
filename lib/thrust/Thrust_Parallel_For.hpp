#ifndef __THRUST_PARALLEL_FOR_HPP__
#define __THRUST_PARALLEL_FOR_HPP__

#include <cassert>
#include <experimental/mdspan>

using counting_iterator = thrust::counting_iterator<int>;

namespace Impl {
  template < class FunctorType >
  void for_each(const int end, const FunctorType f) {
    static_assert( std::is_invocable_v< FunctorType, int > );
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+end,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix) {
                       f(ix);
                     });
  }

  template < class FunctorType >
  void for_each(const int begin, const int end, const FunctorType f) {
    static_assert( std::is_invocable_v< FunctorType, int > );
    const unsigned int n = end - begin;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       const int ix = i + begin;
                       f(ix);
                     });
  }

  template < class FunctorType >
  void for_each(const int1 begin, const int1 end, const FunctorType f) {
    static_assert( std::is_invocable_v< FunctorType, int > );
    const unsigned int n = end.x - begin.x;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       const int ix = i + begin.x;
                       f(ix);
                     });
  }
  
  template < class LayoutPolicy, class FunctorType >
  void for_each( const int2 begin, const int2 end, FunctorType const f ) {
    static_assert( std::is_invocable_v< FunctorType, int, int > );

    const int n0 = end.x - begin.x;
    const int n1 = end.y - begin.y;
    const int n  = n0 * n1;
 
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0 = idx%n0 + begin.x;
                         const int i1 = idx/n0 + begin.y;
                         f(i0, i1);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0 = idx/n0 + begin.x;
                         const int i1 = idx%n0 + begin.y;
                         f(i0, i1);
                       });
    }
  }
  
  template < class LayoutPolicy, class FunctorType >
  void for_each( const int3 begin, const int3 end, const FunctorType f) {
    static_assert( std::is_invocable_v< FunctorType, int, int, int > );
    const int n0 = end.x - begin.x;
    const int n1 = end.y - begin.y;
    const int n2 = end.z - begin.z;
    const int n  = n0 * n1 * n2;
 
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0  = idx % n0 + begin.x;
                         const int i12 = idx / n0;
                         const int i1  = i12%n1 + begin.y;
                         const int i2  = i12/n1 + begin.z;
                         f(i0, i1, i2);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i2  = idx % n2 + begin.z;
                         const int i01 = idx / n2;
                         const int i1  = i01%n1 + begin.y;
                         const int i0  = i01/n1 + begin.x;
                         f(i0, i1, i2);
                       });
    }
  }

  template < class LayoutPolicy, class FunctorType >
  void for_each( const int4 begin, const int4 end, FunctorType const f ) {
    static_assert( std::is_invocable_v< FunctorType, int, int, int, int > );
    const int n0 = end.x - begin.x;
    const int n1 = end.y - begin.y;
    const int n2 = end.z - begin.z;
    const int n3 = end.w - begin.w;
    const int n  = n0 * n1 * n2 * n3;
 
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i0   = idx % n0 + begin.x;
                         const int i123 = idx / n0;
                         const int i1   = i123%n1 + begin.y;
                         const int i23  = i123/n1;
                         const int i2   = i23%n2 + begin.z;
                         const int i3   = i23/n2 + begin.w;
                         f(i0, i1, i2, i3);
                       });
    } else {
      thrust::for_each(thrust::device,
                       counting_iterator(0), counting_iterator(0)+n,
                       [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                         const int i3   = idx % n3 + begin.w;
                         const int i012 = idx / n3;
                         const int i2   = i012%n2 + begin.z;
                         const int i01  = i012/n2;
                         const int i1   = i01%n1 + begin.y;
                         const int i0   = i01/n1 + begin.x;
                         f(i0, i1, i2, i3);
                       });
    }
  }
}

#endif
