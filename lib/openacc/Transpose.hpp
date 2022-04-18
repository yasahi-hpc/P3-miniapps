#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined( ENABLE_OPENACC )
  #include "OpenACC_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

#endif
