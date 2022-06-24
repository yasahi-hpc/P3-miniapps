#ifndef __UTILS_HPP__
#define __UTILS_HPP__

// Class helper
#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
    ClassName(const ClassName&); \
    void operator=(const ClassName&);

#if defined(__CUDACC__)
  inline void synchronize() {
    cudaDeviceSynchronize();
  }
#elif defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  inline void synchronize() {
    [[maybe_unused]] hipError_t err = hipDeviceSynchronize();
  }
#else
  inline void synchronize() {}
#endif

#endif
