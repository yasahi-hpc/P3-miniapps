#ifndef __PARALLEL_FOR_HPP__
#define __PARALLEL_FOR_HPP__

#include <thrust/iterator/counting_iterator.h>
#include <layout_contiguous/layout_contiguous.hpp>
#include "../Iteration.hpp"

using counting_iterator = thrust::counting_iterator<int>;

#if defined( ENABLE_CUDA ) && ! defined( ENABLE_THRUST )
  /* policies if you choose CUDA backend
   */
  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start, int end, const FunctorType f) {
    for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin; ix<end; ix+=gridDim.x*blockDim.x) {
      f(ix);
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin[1]; iy<end[1]; iy+=gridDim.y*blockDim.y) {
      for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin[0]; ix<end[0]; ix+=gridDim.x*blockDim.x) {
        f(ix, iy);
      }
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    for(int iz=blockIdx.z*blockDim.z+threadIdx.z+begin[2]; iz<end[2]; iz+=gridDim.z*blockDim.z) {
      for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin[1]; iy<end[1]; iy+=gridDim.y*blockDim.y) {
        for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin[0]; ix<end[0]; ix+=gridDim.x*blockDim.x) {
          f(ix, iy, iz);
        }
      }
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    const int n2 = end[2] - start[2], n3 = end[3] - start[3];
    const int n23 = n2 * n3;
    for(int izw=blockIdx.z*blockDim.z+threadIdx.z; izw<n23; izw+=gridDim.z*blockDim.z) {
      const int iz = izw % n2 + start[2];
      const int iw = izw / n2 + start[3];
      for(int iy=blockIdx.y*blockDim.y+threadIdx.y+begin[1]; iy<end[1]; iy+=gridDim.y*blockDim.y) {
        for(int ix=blockIdx.x*blockDim.x+threadIdx.x+begin[0]; ix<end[0]; ix+=gridDim.x*blockDim.x) {
          f(ix, iy, iz, iw);
        }
      }
    }
  }

  namespace Impl {
    constexpr size_t nb_threads = 256;
    constexpr size_t nb_threads_x = 32, nb_threads_y = 8;

    
    template <class IteratePolicy, class FunctorType,
              std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
    void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
      static_assert( IteratePolicy::rank() == 1 );
      auto n = iterate_policy.size();
      const auto start = iterate_policy.start();
      const auto start0 = start[0], end0 = start0 + n;
      const dim3 blocks(nb_threads, 1, 1);
      const dim3 grids( (n-1) / blocks.x + 1, 1, 1);

      kernel_for_each<<<grids, blocks>>>(start0, end0, std::forward<FunctorType>(f));
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
      const int start_[2] = {start0, start1};
      const int end_[2] = {start0+n0, start1+n1};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, 1);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        kernel_for_each<<<grids, blocks>>>(start_, end_, std::forward<FunctorType>(f));
      }
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
      const int start_[3] = {start0, start1, start2};
      const int end_[3] = {start0+n0, start1+n1, start2+n2};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        kernel_for_each<<<grids, blocks>>>(start_, end_, std::forward<FunctorType>(f));
      }
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
      const int start_[3] = {start0, start1, start2, start3};
      const int end_[3] = {start0+n0, start1+n1, start2+n2, start3+n3};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2 * n3);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        kernel_for_each<<<grids, blocks>>>(start_, end_, std::forward<FunctorType>(f));
      }
    }
  };
#elif defined( ENABLE_HIP ) && ! defined( ENABLE_THRUST )
  /* Default policies for hip 
   */
  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start, int end, const FunctorType f) {
    for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+start; ix<end; ix+=hipGridDim_x*hipBlockDim_x) {
      f(ix);
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+start[1]; ix<end[1]; ix+=hipGridDim_y*hipBlockDim_y) {
      for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+start[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
        f(ix, iy);
      }
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    for(int iz=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z+start[2]; iz<end[2]; iz+=hipGridDim_z*hipBlockDim_z) {
      for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+start[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
        for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+start[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
          f(ix, iy, iz);
        }
      }
    }
  }

  template < class FunctorType, std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr> 
  __global__ void kernel_for_each(int start[], int end[], const FunctorType f) {
    const int n2 = end[2] - start[2], n3 = end[3] - start[3];
    const int n23 = n2 * n3;
    for(int izw=hipBlockIdx_z*hipBlockDim_z+hipThreadIdx_z+start[2]; izw<end[2]; izw+=hipGridDim_z*hipBlockDim_z) {
      const int iz = izw % n2 + start[2];
      const int iw = izw / n2 + start[3];
      for(int iy=hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y+start[1]; iy<end[1]; iy+=hipGridDim_y*hipBlockDim_y) {
        for(int ix=hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x+start[0]; ix<end[0]; ix+=hipGridDim_x*hipBlockDim_x) {
          f(ix, iy, iz, iw);
        }
      }
    }
  }

  namespace Impl {
    constexpr size_t nb_threads = 256;
    constexpr size_t nb_threads_x = 32, nb_threads_y = 8;

    
    template <class IteratePolicy, class FunctorType,
              std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
    void for_each(const IteratePolicy iterate_policy, const FunctorType f) {
      static_assert( IteratePolicy::rank() == 1 );
      auto n = iterate_policy.size();
      const auto start = iterate_policy.start();
      const auto start0 = start[0], end0 = start0 + n;
      const dim3 blocks(nb_threads, 1, 1);
      const dim3 grids( (n-1) / blocks.x + 1, 1, 1);

      hipLaunchKernelGGL(kernel_for_each, grids, blocks, 0, 0, start0, end0, std::forward<FunctorType>(f));
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
      const int start_[2] = {start0, start1};
      const int end_[2] = {start0+n0, start1+n1};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, 1);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        hipLaunchKernelGGL(kernel_for_each, grids, blocks, 0, 0, start_, end_, std::forward<FunctorType>(f));
      }
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
      const int start_[3] = {start0, start1, start2};
      const int end_[3] = {start0+n0, start1+n1, start2+n2};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        hipLaunchKernelGGL(kernel_for_each, grids, blocks, 0, 0, start_, end_, std::forward<FunctorType>(f));
      }
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
      const int start_[3] = {start0, start1, start2, start3};
      const int end_[3] = {start0+n0, start1+n1, start2+n2, start3+n3};
      const dim3 blocks(nb_threads_x, nb_threads_y, 1);
      const dim3 grids( (n0-1) / blocks.x + 1, (n1-1) / blocks.y + 1, n2 * n3);
    
      if(std::is_same_v<typename IteratePolicy::layout_type, layout_contiguous_at_left>) {
        hipLaunchKernelGGL(kernel_for_each, grids, blocks, 0, 0, start_, end_, std::forward<FunctorType>(f));
      }
    }
  };
#else
  /* Default policies for thrust
   */
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
    }
  };
#endif

#endif
