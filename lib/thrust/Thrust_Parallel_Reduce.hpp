#ifndef __THRUST_PARALLEL_REDUCE_HPP__
#define __THRUST_PARALLEL_REDUCE_HPP__

#include <cassert>
#include <experimental/mdspan>

using counting_iterator = thrust::counting_iterator<int>;

namespace Impl {

  template < class OutputType, class UnarayOperation, class BinaryOperation >
  void transform_reduce( const int1 begin, const int1 end, 
                         BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result) {
    const unsigned int n = end.x - begin.x;
    OutputType tmp = result;

    result = thrust::transform_reduce(thrust::device,
                                      counting_iterator(0), counting_iterator(0)+n,
                                      [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                        const int ix = i + begin.x;
                                        return unary_op(ix);
                                      },
                                      tmp,
                                      binary_op
                                     );
  }
  
  template < class LayoutPolicy, class OutputType, class UnarayOperation, class BinaryOperation >
  void transform_reduce( const int2 begin, const int2 end, 
                         BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int n  = nx * ny;
    OutputType tmp = result;

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int ix = i%nx + begin.x;
                                          const int iy = i/nx + begin.y;
                                          return unary_op(ix, iy);
                                        },
                                        tmp,
                                        binary_op
                                       );
    } else {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int ix = i/nx + begin.x;
                                          const int iy = i%nx + begin.y;
                                          return unary_op(ix, iy);
                                        },
                                        tmp,
                                        binary_op
                                       );
    }
  }
  
  //template < class OutputType, class UnarayOperation, class BinaryOperation, class LayoutPolicy,
  //           std::enable_if_t<std::is_invocable_v< UnarayOperation, int, int, int > && std::is_invocable_v< BinaryOperation, OutputType&, OutputType& >, std::nullptr_t> = nullptr>
  template < class LayoutPolicy, class OutputType, class UnarayOperation, class BinaryOperation >
  void transform_reduce( const int3 begin, const int3 end,
                         BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;
    const int n  = nx * ny * nz;
    OutputType tmp = result;

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int ix = i%nx + begin.x;
                                          const int iyz = i/nx;
                                          const int iy  = iyz%ny + begin.y;
                                          const int iz  = iyz/ny + begin.z;
                                          return unary_op(ix, iy, iz);
                                        },
                                        tmp,
                                        binary_op
                                       );
    } else {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int iz = i%nz + begin.z;
                                          const int ixy = i/nz;
                                          const int iy  = ixy%ny + begin.y;
                                          const int ix  = ixy/ny + begin.x;
                                          return unary_op(ix, iy, iz);
                                        },
                                        tmp,
                                        binary_op
                                       );
    }
  }

  //template < class OutputType, class UnarayOperation, class BinaryOperation, class LayoutPolicy,
  //           std::enable_if_t<std::is_invocable_v< UnarayOperation, int, int, int, int > && std::is_invocable_v< BinaryOperation, OutputType&, OutputType& >, std::nullptr_t> = nullptr>
  template < class LayoutPolicy, class OutputType, class UnarayOperation, class BinaryOperation >
  void transform_reduce( const int4 begin, const int4 end,
                         BinaryOperation const binary_op, UnarayOperation const unary_op, OutputType &result) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;
    const int nw = end.w - begin.w;
    const int n  = nx * ny * nz * nw;
    OutputType tmp = result;

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int ix = i%nx + begin.x;
                                          const int iyzw = i/nx;
                                          const int iy  = iyzw%ny + begin.y;
                                          const int izw = iyzw/ny;
                                          const int iz  = izw%nz + begin.z;
                                          const int iw  = izw/nz + begin.w;
                                          return unary_op(ix, iy, iz, iw);
                                        },
                                        tmp,
                                        binary_op
                                       );
    } else {
      result = thrust::transform_reduce(thrust::device,
                                        counting_iterator(0), counting_iterator(0)+n,
                                        [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                                          const int iw = i%nw + begin.w;
                                          const int ixyz = i/nw;
                                          const int iz  = ixyz%nz + begin.z;
                                          const int ixy = ixyz/nz;
                                          const int iy  = ixy%ny + begin.y;
                                          const int ix  = ixy/ny + begin.x;
                                          return unary_op(ix, iy, iz, iw);
                                        },
                                        tmp,
                                        binary_op
                                       );
    }
  }
}

#endif
