#ifndef __OPENMP_PARALLEL_FOR_HPP__
#define __OPENMP_PARALLEL_FOR_HPP__

#include <cassert>
#include <experimental/mdspan>

namespace Impl {

  template < class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
  void for_each( const int1 begin, const int1 end, FunctorType const f ) {
    const unsigned int n = end.x - begin.x;

    #pragma omp parallel for
    for(int i=0; i<n; i++) {
      const int ix = i + begin.x;
      std::forward<FunctorType>(f)(ix);
    }
  }
  
  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr>
  void for_each( const int2 begin, const int2 end, FunctorType const f ) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      #pragma omp parallel for 
      for(int j=0; j<ny; j++) {
        const int iy = j + begin.y;
        #pragma omp simd
        for(int i=0; i<nx; i++) {
          const int ix = i + begin.x;
          std::forward<FunctorType>(f)(ix, iy);
        }
      }
    } else {
      #pragma omp parallel for 
      for(int i=0; i<nx; i++) {
        const int ix = i + begin.x;
        #pragma omp simd
        for(int j=0; j<ny; j++) {
          const int iy = j + begin.y;
          std::forward<FunctorType>(f)(ix, iy);
        }
      }
    }
  }
  
  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr>
  void for_each( const int3 begin, const int3 end, FunctorType const f ) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      #pragma omp parallel for collapse(2)
      for(int k=0; k<nz; k++) {
        for(int j=0; j<ny; j++) {
          const int iz = k + begin.z;
          const int iy = j + begin.y;
          #pragma omp simd
          for(int i=0; i<nx; i++) {
            const int ix = i + begin.x;
            std::forward<FunctorType>(f)(ix, iy, iz);
          }
        }
      }
    } else {
      #pragma omp parallel for collapse(2)
      for(int i=0; i<nx; i++) {
        for(int j=0; j<ny; j++) {
          const int ix = x + begin.x;
          const int iy = j + begin.y;
          #pragma omp simd
          for(int k=0; k<nz; k++) {
            const int iz = k + begin.z;
            std::forward<FunctorType>(f)(ix, iy, iz);
          }
        }
      }
    }
  }

  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr>
  void for_each( const int4 begin, const int4 end,  FunctorType const f ) {
    const int nx = end.x - begin.x;
    const int ny = end.y - begin.y;
    const int nz = end.z - begin.z;
    const int nw = end.w - begin.w;
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      #pragma omp parallel for collapse(3)
      for(int l=0; l<nw; l++) {
        for(int k=0; k<nz; k++) {
          for(int j=0; j<ny; j++) {
            const int iw = l + begin.w;
            const int iz = k + begin.z;
            const int iy = j + begin.y;
            #pragma omp simd
            for(int i=0; i<nx; i++) {
              const int ix = i + begin.x;
              std::forward<FunctorType>(f)(ix, iy, iz, iw);
            }
          }
        }
      }
    } else {
      #pragma omp parallel for collapse(3)
      for(int i=0; i<nx; i++) {
        for(int j=0; j<ny; j++) {
          for(int k=0; k<nw; k++) {
          const int ix = x + begin.x;
          const int iy = j + begin.y;
          const int iz = k + begin.z;
          #pragma omp simd
          for(int l=0; l<nw; l++) {
            const int iw = l + begin.w;
            std::forward<FunctorType>(f)(ix, iy, iz, iw);
          }
        }
      }
    }
  }
}

#endif
