#ifndef __CUDA_PARALLEL_FOR_HPP__
#define __CUDA_PARALLEL_FOR_HPP__

#include <cassert>
#include <experimental/mdspan>

// Layout does not matter for 1D case
template < class FunctorType >
__global__ void kernel_for_each(int begin, int end, const FunctorType f) {
  static_assert( std::is_invocable_v< FunctorType, int > );
  for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin; ix<end; ix+=gridDim.x*blockDim.x) {
    f(ix);
  }
}

// 2D case
template < class FunctorType >
__global__ void kernel_for_each_left(int2 begin, int2 end, const FunctorType f) {
  for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
    for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
      f(ix, iy);
    }
  }
}

template < class FunctorType >
__global__ void kernel_for_each_right(int2 begin, int2 end, const FunctorType f) {
  for(int ix=blockIdx.y*blockDim.y+threadIdx.y+begin.y; ix<end.y; ix+=gridDim.y*blockDim.y) {
    for(int iy=blockIdx.x*blockDim.x+threadIdx.x+begin.x; iy<end.x; iy+=gridDim.x*blockDim.x) {
      f(ix, iy);
    }
  }
}

// 3D case
template < class FunctorType >
__global__ void kernel_for_each_left(int3 begin, int3 end, const FunctorType f) {
  for(int iz=blockIdx.z*blockDim.z+threadIdx.z+begin.z; iz<end.z; iz+=gridDim.z*blockDim.z) {
    for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
      for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
        f(ix, iy, iz);
      }
    }
  }
}

template < class FunctorType >
__global__ void kernel_for_each_right(int3 begin, int3 end, const FunctorType f) {
  for(int ix=blockIdx.z*blockDim.z+threadIdx.z+begin.z; ix<end.z; ix+=gridDim.z*blockDim.z) {
    for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
      for(int iz=blockIdx.x*blockDim.x+threadIdx.x+begin.x; iz<end.x; iz+=gridDim.x*blockDim.x) {
        f(ix, iy, iz);
      }
    }
  }
}

// 4D case
template < class FunctorType >
__global__ void kernel_for_each_left(int4 begin, int4 end, const FunctorType f) {
  const int n2 = end.z - begin.z, n3 = end.w - begin.w;
  const int n23 = n2 * n3;
  for(int izw=blockIdx.z*blockDim.z+threadIdx.z; izw<n23; izw+=gridDim.z*blockDim.z) {
    const int iz = izw % n2 + begin.z;
    const int iw = izw / n2 + begin.w;
    for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
      for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
        f(ix, iy, iz, iw);
      }
    }
  }
}

template < class FunctorType >
__global__ void kernel_for_each_right(int4 begin, int4 end, const FunctorType f) {
  const int n2 = end.z - begin.z, n3 = end.w - begin.w;
  const int n23 = n2 * n3;
  for(int ixy=blockIdx.z*blockDim.z+threadIdx.z; ixy<n23; ixy+=gridDim.z*blockDim.z) {
    const int iy = ixy % n2 + begin.z;
    const int ix = ixy / n2 + begin.w;
    for(int iz=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iz<end.y; iz+=gridDim.y*blockDim.y) {
      for(int iw=blockIdx.x*blockDim.x+threadIdx.x+begin.x; iw<end.x; iw+=gridDim.x*blockDim.x) {
        f(ix, iy, iz, iw);
      }
    }
  }
}

namespace Impl {
  // Default thread size
  constexpr size_t nb_threads = 256;
  constexpr size_t nb_threads_x = 32, nb_threads_y = 8;

  template < class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr >
  void for_each( const int1 begin, const int1 end, FunctorType && f, const size_t _nb_threads=nb_threads ) {
    int begin0 = begin.x, end0 = end.x;
    int n = end0 - begin0;
    const dim3 blocks(_nb_threads, 1, 1);
    const dim3 grids( (n-1) / blocks.x + 1, 1, 1);

    kernel_for_each<<<grids, blocks>>>(begin0, end0, std::forward<FunctorType>(f));
  }
  
  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int2 begin, const int2 end, FunctorType && f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {

    int2 begin_, end_;
    int n0, n1;
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      n0 = end.x - begin.x;
      n1 = end.y - begin.y;
      begin_ = make_int2(begin.x, begin.y);
      end_   = make_int2(begin.x+n0, begin.y+n1);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, 1);
      kernel_for_each_left<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    } else {
      n0 = end.y - begin.y;
      n1 = end.x - begin.x;
      begin_ = make_int2(begin.y, begin.x);
      end_   = make_int2(begin.y+n0, begin.x+n1);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, 1);
      kernel_for_each_right<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    }

  }
  
  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int3 begin, const int3 end, FunctorType && f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {
    int3 begin_, end_;
    int n0, n1, n2;
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      n0 = end.x - begin.x;
      n1 = end.y - begin.y;
      n2 = end.z - begin.z;
      begin_ = make_int3(begin.x, begin.y, begin.z);
      end_ = make_int3(begin.x+n0, begin.y+n1, begin.z+n2);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2);
      kernel_for_each_left<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    } else {
      n0 = end.z - begin.z;
      n1 = end.y - begin.y;
      n2 = end.x - begin.x;
      begin_ = make_int3(begin.z, begin.y, begin.x);
      end_ = make_int3(begin.z+n0, begin.y+n1, begin.x+n2);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2);
      kernel_for_each_right<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    }
  }

  template < class LayoutPolicy, class FunctorType,
             std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr >
  void for_each( const int4 begin, const int4 end, FunctorType && f,
                 const size_t _nb_threads_x=nb_threads_x, const size_t _nb_threads_y=nb_threads_y ) {
    int4 begin_, end_;
    int n0, n1, n2, n3;
    if(std::is_same_v<LayoutPolicy, stdex::layout_left>) {
      n0 = end.x - begin.x;
      n1 = end.y - begin.y;
      n2 = end.z - begin.z;
      n3 = end.w - begin.w;

      begin_ = make_int4(begin.x, begin.y, begin.z, begin.w);
      end_ = make_int4(begin.x+n0, begin.y+n1, begin.z+n2, begin.w+n3);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2*n3);
      kernel_for_each_left<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    } else {
      n0 = end.w - begin.w;
      n1 = end.z - begin.z;
      n2 = end.y - begin.y;
      n3 = end.x - begin.x;

      begin_ = make_int4(begin.w, begin.z, begin.y, begin.x);
      end_ = make_int4(begin.w+n0, begin.z+n1, begin.y+n2, begin.x+n3);
      const dim3 blocks(_nb_threads_x, _nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2*n3);
      kernel_for_each_right<<<grids, blocks>>>(begin_, end_, std::forward<FunctorType>(f));
    }

  }
}

#endif
