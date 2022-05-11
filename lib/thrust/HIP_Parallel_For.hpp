#ifndef __HIP_PARALLEL_FOR_HPP__
#define __HIP_PARALLEL_FOR_HPP__

#include <cassert>
#include <experimental/mdspan>

// Layout does not matter for 1D case
template < class FunctorType, class LayoutPolicy=stdex::layout_left,
           std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start, int end, const FunctorType f) {
  for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin; ix<end; ix+=hipGridDim_x*hipBlockDim_x) {
    f(ix);
  }
}

// 2D case
template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_left>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
    for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
      f(ix, iy);
    }
  }
}

template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_right>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  for(int ix=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; ix<end[1]; ix+=hipGridDim_y*hipBlockDim_y) {
    for(int iy=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; iy<end[0]; iy+=hipGridDim_x*hipBlockDim_x) {
      f(ix, iy);
    }
  }
}

// 3D case
template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_left>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  for(int iz=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z+begin[2]; iz<end[2]; iz+=hipGridDim_z*hipBlockDim_z) {
    for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
      for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
        f(ix, iy, iz);
      }
    }
  }
}

template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_right>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  for(int ix=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z+begin[2]; ix<end[2]; ix+=hipGridDim_z*hipBlockDim_z) {
    for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
      for(int iz=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; iz<end[0]; iz+=hipGridDim_x*hipBlockDim_x) {
        f(ix, iy, iz);
      }
    }
  }
}

// 4D case
template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_left>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  const int n2 = end[2] - start[2], n3 = end[3] - start[3];
  const int n23 = n2 * n3;
  for(int izw=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z; izw<n23; izw+=hipGridDim_z*hipBlockDim_z) {
    const int iz = izw % n2 + start[2];
    const int iw = izw / n2 + start[3];
    for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
      for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
        f(ix, iy, iz, iw);
      }
    }
  }
}

template < class FunctorType, class LayoutPolicy,
           std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int > && std::is_same_v<LayoutPolicy, stdex::layout_right>, std::nullptr_t> = nullptr>
__global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
  const int n2 = end[2] - start[2], n3 = end[3] - start[3];
  const int n23 = n2 * n3;
  for(int ixy=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z; ixy<n23; ixy+=hipGridDim_z*hipBlockDim_z) {
    const int iy = ixy % n2 + start[2];
    const int ix = ixy / n2 + start[3];
    for(int iz=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+begin[1]; iz<end[1]; iz+=hipGridDim_y*hipBlockDim_y) {
      for(int iw=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+begin[0]; iw<end[0]; iw+=hipGridDim_x*hipBlockDim_x) {
        f(ix, iy, iz, iw);
      }
    }
  }
}

namespace Impl {
  // Default thread size
  constexpr size_t nb_threads = 256;
  constexpr size_t nb_threads_x = 32, nb_threads_y = 8;

  template < class F, class LayoutPolicy,
             std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr >
  void for_each( const int1 begin, const int1 end, F const f, const size_t _nb_threads=nb_threads ) {
    const dim3 blocks(_nb_threads, 1, 1);
    const dim3 grids( (n-1) / blocks.x + 1, 1, 1);
    int begin0 = begin.x, end0 = end.x;

    hipLaunchKernelGGL(kernel_for_each, grids, blocks, 0, 0, begin0, end0, std::forward<FunctorType>(f));
  }
  
  template < class F, class LayoutPolicy,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int2 begin, const int2 end, F const f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {

    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      const int n0 = end.x - begin.x;
      const int n1 = end.y - begin.y;
      const int begin_[2] = {begin.x, begin.y};
      const int end_[2] = {begin.x+n0, begin.y+n1};
    } else {
      const int n0 = end.y - begin.y;
      const int n1 = end.x - begin.x;
      const int begin_[2] = {begin.y, begin.x};
      const int end_[2] = {begin.y+n0, begin.x+n1};
    }

    const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
    const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, 1);

    hipLaunchKernelGGL(kernel_for_each<LayoutPolicy>, grids, blocks, 0, 0, begin_, end_, std::forward<FunctorType>(f));
  }
  
  template < class F, class LayoutPolicy,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int3 begin, const int3 end, F const f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      const int n0 = end.x - begin.x;
      const int n1 = end.y - begin.y;
      const int n2 = end.z - begin.z;
      const int begin_[3] = {begin.x, begin.y, begin.z};
      const int end_[3] = {begin.x+n0, begin.y+n1, begin.z+n2};
    } else {
      const int n0 = end.z - begin.z;
      const int n1 = end.y - begin.y;
      const int n2 = end.x - begin.x;
      const int begin_[3] = {begin.z, begin.y, begin.x};
      const int end_[3] = {begin.z+n0, begin.y+n1, begin.x+n2};
    }

    const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
    const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2);

    hipLaunchKernelGGL(kernel_for_each<LayoutPolicy>, grids, blocks, 0, 0, begin_, end_, std::forward<FunctorType>(f));
  }

  template < class F, class LayoutPolicy,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int4 begin, const int4 end, F const f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      const int n0 = end.x - begin.x;
      const int n1 = end.y - begin.y;
      const int n2 = end.z - begin.z;
      const int n3 = end.w - begin.w;

      const int begin_[4] = {begin.x, begin.y, begin.z, begin.w};
      const int end_[4] = {begin.x+n0, begin.y+n1, begin.z+n2, begin.w+n3};
    } else {
      const int n0 = end.w - begin.w;
      const int n1 = end.z - begin.z;
      const int n2 = end.y - begin.y;
      const int n3 = end.x - begin.x;

      const int begin_[4] = {begin.w, begin.z, begin.y, begin.x};
      const int end_[4] = {begin.w+n0, begin.z+n1, begin.y+n2, begin.x+n3};
    }

    const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
    const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2*n3);

    hipLaunchKernelGGL(kernel_for_each<LayoutPolicy>, grids, blocks, 0, 0, begin_, end_, std::forward<FunctorType>(f));
  }
}

#endif
