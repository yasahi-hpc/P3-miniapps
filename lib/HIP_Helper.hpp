#ifndef __HIP_HELPER_HPP__
#define __HIP_HELPER_HPP__

#include <cstdlib>
#include <string>
#include <hip/hip_runtime.h>

#define SafeHIPCall(call) CheckHIPCall(call, #call, __FILE__, __LINE__)

template <typename T>
void CheckHIPCall(T command, const char * commandName, const char * fileName, int line) {
  if(command) {
    fprintf(stderr, "HIP error at %s:%d code=%d \"%s\" \n",
            fileName, line, (unsigned int)command, commandName);

    [[maybe_unused]] hipError_t err = hipDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#endif
