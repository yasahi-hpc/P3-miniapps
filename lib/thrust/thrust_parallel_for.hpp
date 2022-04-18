#ifndef __STDPAR_KERNELS_HPP__
#define __STDPAR_KERNELS_HPP__

#include <cassert>

using counting_iterator = thrust::counting_iterator<int>;

namespace Impl {

  template < class F >
  void for_each( const int1 begin, const int1 end, F const f ) {
    static_assert( std::is_invocable_v< F, int > );
    const unsigned int n = end.x - begin.x;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       const int ix = i + begin.x;
                       f(ix);
                     });
  }
  
  template < class F >
  void for_each( const int2 begin, const int2 end, F const f ) {
    static_assert( std::is_invocable_v< F, int, int > );
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int n  = nx * ny;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       #if defined( ENABLE_CUDA) || defined( ENABLE_HIP )
                         const int ix = i%nx + begin.x;
                         const int iy = i/nx + begin.y;
                       #else
                         const int ix = i/nx + begin.x;
                         const int iy = i%nx + begin.y;
                       #endif
                       f(ix, iy);
                     });
  }
  
  template < class F >
  void for_each( const int3 begin, const int3 end, F const f ) {
    static_assert( std::is_invocable_v< F, int, int, int > );
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;
    const int n  = nx * ny * nz;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       #if defined( ENABLE_CUDA) || defined( ENABLE_HIP )
                         const int ix = i%nx + begin.x;
                         const int iyz = i/nx;
                         const int iy  = iyz%ny + begin.y;
                         const int iz  = iyz/ny + begin.z;
                       #else
                         const int iz = i%nz + begin.z;
                         const int ixy = i/nz;
                         const int iy  = ixy%ny + begin.y;
                         const int ix  = ixy/ny + begin.x;
                       #endif
                       f(ix, iy, iz);
                     });
  }

  template < class F >
  void for_each( const int4 begin, const int4 end, F const f ) {
    static_assert( std::is_invocable_v< F, int, int, int, int > );
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;
    const int nw = end.w - begin.w;
    const int n  = nx * ny * nz * nw;
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i) {
                       #if defined( ENABLE_CUDA) || defined( ENABLE_HIP )
                         const int ix = i%nx + begin.x;
                         const int iyzw = i/nx;
                         const int iy  = iyzw%ny + begin.y;
                         const int izw = iyzw/ny;
                         const int iz  = izw%nz + begin.z;
                         const int iw  = izw/nz + begin.w;
                       #else
                         const int iw = i%nw + begin.w;
                         const int ixyz = i/nw;
                         const int iz  = ixyz%nz + begin.z;
                         const int ixy = ixyz/nz;
                         const int iy  = ixy%ny + begin.y;
                         const int ix  = ixy/ny + begin.x;
                       #endif
                       f(ix, iy, iz, iw);
                     });
  }
}

#endif
