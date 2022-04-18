#ifndef __PARALLEL_FOR_EACH_HPP__
#define __PARALLEL_FOR_EACH_HPP__

#include <cassert>

template < class F >
__global__ void kernel_for_each(const uint1 begin, const uint1 end, const F f) {
  static_assert( std::is_invocable_v< F, unsigned > );
  for(unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
    f(ix);
  }
}

template < class F >
__global__ void kernel_for_each(const uint2 begin, const uint2 end, const F f) {
  static_assert( std::is_invocable_v< F, unsigned, unsigned > );
  for(unsigned int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
    for(unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
      f(ix, iy);
    }
  }
}

template < class F >
__global__ void kernel_for_each(const uint3 begin, const uint3 end, const F f) {
  static_assert( std::is_invocable_v< F, unsigned, unsigned, unsigned > );
  for(unsigned int iz=blockIdx.z*blockDim.z+threadIdx.z+begin.z; iz<end.z; iz+=gridDim.z*blockDim.z) {
    for(unsigned int iy=blockIdx.y*blockDim.y+threadIdx.y+begin.y; iy<end.y; iy+=gridDim.y*blockDim.y) {
      for(unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x+begin.x; ix<end.x; ix+=gridDim.x*blockDim.x) {
        f(ix, iy, iz);
      }
    }
  }
}

template < class F >
void parallel_for_each( const uint1 begin, const uint1 end, F&& f ) {
  static_assert( std::is_invocable_v< F, unsigned > );

  const unsigned int n = end.x - begin.x;
  const dim3 blocks(128, 1, 1);
  const dim3 grids( (n-1) / blocks.x + 1, 1, 1);
  kernel_for_each<<<grids, blocks>>>(begin, end, std::forward<F>(f));
}

template < class F >
void parallel_for_each( const uint2 begin, const uint2 end, F&& f ) {
  static_assert( std::is_invocable_v< F, unsigned, unsigned > );

  const unsigned int nx = end.x - begin.x;
  const unsigned int ny = end.y - begin.y;
  const dim3 blocks(32, 8, 1);
  const dim3 grids( (nx-1) / blocks.x + 1, (ny-1) / blocks.y + 1, 1);
  kernel_for_each<<<grids, blocks>>>(begin, end, std::forward<F>(f));
}

template < class F >
void parallel_for_each( const uint3 begin, const uint3 end, F&& f ) {
  static_assert( std::is_invocable_v< F, unsigned, unsigned, unsigned > );

  const unsigned int nx = end.x - begin.x;
  const unsigned int ny = end.y - begin.y;
  const unsigned int nz = end.z - begin.z;
  const dim3 blocks(32, 8, 1);
  const dim3 grids( (nx-1) / blocks.x + 1, (ny-1) / blocks.y + 1, (nz-1) / blocks.z + 1 );
  kernel_for_each<<<grids, blocks>>>(begin, end, std::forward<F>(f));
}

#endif
